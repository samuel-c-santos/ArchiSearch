import os
import faiss
import numpy as np
import torch
import base64
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from flask import Flask, request, render_template, jsonify
from tqdm import tqdm
from io import BytesIO
import subprocess

# =========================================================================
# 1. CONFIGURAÇÕES E CONSTANTES
# =========================================================================

# --- AJUSTE ESTE CAMINHO PARA O SEU REPOSITÓRIO DE IMAGENS ---
# Mantenha o formato de string crua (r"...") para evitar problemas com backslashes no Windows
IMAGE_ROOT_DIR = r"H:\Meu Drive\EDIFICAÇÕES\IDEIAS" 
# COMENTÁRIO: Substitua o caminho acima pelo seu caminho real
# -------------------------------------------------------------

INDEX_FILE = "architecture_faiss.index"
MAPPING_FILE = "architecture_mapping.npy"
MODEL_NAME = "openai/clip-vit-base-patch32"

# Variáveis globais para o modelo e o índice
model = None
processor = None
device = "cpu"
index = None
mapping = None


# CLASSE DE CORREÇÃO CRÍTICA: Codificador JSON customizado para lidar com tipos NumPy/PyTorch
class NpEncoder(json.JSONEncoder):
    """Lida com tipos numpy.float32 e numpy.float64 para que possam ser serializados em JSON."""
    def default(self, obj):
        # Converte tipos float do NumPy (como float32, float64) para o float nativo do Python
        if isinstance(obj, np.floating):
            return float(obj)
        # Converte tipos int do NumPy para o int nativo do Python
        if isinstance(obj, np.integer):
            return int(obj)
        # Deixe o codificador padrão do JSON lidar com o restante
        return json.JSONEncoder.default(self, obj)


# Configuração do aplicativo Flask
app = Flask(__name__)
# APLICA A CORREÇÃO: Define o codificador customizado para o Flask
# Isso resolve o erro "Object of type float32 is not JSON serializable"
app.json_encoder = NpEncoder


# =========================================================================
# 2. FUNÇÕES DE SETUP E CARREGAMENTO (Indexação)
# =========================================================================

def load_model():
    """Carrega o modelo CLIP e o processador."""
    global model, processor, device
    print("Iniciando carregamento do modelo CLIP...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        print(f"Modelo CLIP carregado com sucesso. Usando dispositivo: {device}")
        return True
    except Exception as e:
        print(f"ERRO ao carregar o modelo CLIP: {e}")
        return False

def get_image_paths(root_dir):
    """Retorna uma lista de caminhos de imagens suportadas."""
    print(f"Procurando imagens em: {root_dir}")
    allowed_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.tiff')
    image_paths = []
    
    # Percorre o diretorio e subdiretorios
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(allowed_extensions):
                image_paths.append(os.path.join(dirpath, filename))
                
    return image_paths

def generate_embeddings(image_paths, model, processor, device):
    """Gera o embedding (vetor) para cada imagem."""
    embeddings = []
    processed_paths = []
    batch_size = 32

    # Usa tqdm apenas no servidor para logs, não para UI interativa
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processando Imagens"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        # 1. Carrega as imagens para o lote
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
                processed_paths.append(path)
            except Exception:
                pass
                
        if not batch_images:
            continue

        # 2. Processa e gera os embeddings
        try:
            inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            
            # Normalização crucial para similaridade por cosseno
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embeddings.append(image_features.cpu().numpy())
        except Exception as e:
            print(f"\nERRO ao gerar embeddings para um lote. Erro: {e}")

    if not embeddings:
        return None, None

    final_embeddings = np.concatenate(embeddings, axis=0).astype('float32')
    return final_embeddings, processed_paths

def create_faiss_index(embeddings, processed_paths):
    """Cria e salva o indice FAISS e o mapeamento de caminhos."""
    if embeddings is None:
        return None, None

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Salva os arquivos de índice
    faiss.write_index(index, INDEX_FILE)
    np.save(MAPPING_FILE, processed_paths)
    
    print(f"Indice FAISS salvo em {INDEX_FILE}. Total de {index.ntotal} itens.")
    return index, processed_paths

def load_index_and_mapping():
    """Tenta carregar o indice FAISS e o mapeamento existentes."""
    global index, mapping
    
    # Verifica explicitamente a existencia do arquivo antes de tentar carregar
    if not os.path.exists(INDEX_FILE) or not os.path.exists(MAPPING_FILE):
        print("Arquivos de indice FAISS nao encontrados. O sistema precisa ser indexado.")
        index = None
        mapping = None
        return False
        
    try:
        index = faiss.read_index(INDEX_FILE)
        # O allow_pickle=True é necessario para carregar listas de strings
        mapping = np.load(MAPPING_FILE, allow_pickle=True).tolist()
        print(f"Indice FAISS e mapeamento carregados. Total de {index.ntotal} itens indexados.")
        return True
    except RuntimeError as e:
        # Captura o erro do FAISS caso o arquivo exista mas esteja corrompido ou inacessível por algum motivo.
        print(f"ERRO FATAL ao carregar o indice FAISS: {e}")
        index = None
        mapping = None
        return False

# =========================================================================
# 3. FUNÇÕES DE BUSCA
# =========================================================================

def get_vector_from_input(data, model, processor, device):
    """Converte o input (texto ou imagem base64) em um vetor."""
    if data.get('query_type') == 'text':
        text = data.get('query_value')
        if not text:
            raise ValueError("Consulta de texto vazia.")
            
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            
    elif data.get('query_type') == 'image':
        base64_img = data.get('query_value')
        if not base64_img:
            raise ValueError("Imagem de referencia nao fornecida.")

        # Converte base64 para imagem PIL
        # Tenta lidar com diferentes formatos de prefixo data url
        if ',' in base64_img:
            image_bytes = base64_img.split(',', 1)[1] 
        else:
            image_bytes = base64_img
            
        img = Image.open(BytesIO(base64.b64decode(image_bytes))).convert("RGB")
        
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
    
    else:
        raise ValueError("Tipo de consulta invalido.")
    
    # Normalização
    features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype('float32')

def search_index(query_vector, index, mapping, k=5):
    """Executa a busca e retorna caminho + imagem em Base64."""
    if index is None or mapping is None:
        return []

    D, indices = index.search(query_vector, k) 
    
    results = []
    for i in range(min(k, len(indices[0]))): 
        # 1. Cálculos básicos
        raw_similarity = 1 - (D[0][i] / 2) 
        similarity = float(raw_similarity)
        file_path = str(mapping[indices[0][i]])
        
        # 2. Gerar miniatura em Base64
        img_base64 = None
        try:
            # Abre a imagem
            if os.path.exists(file_path):
                img = Image.open(file_path).convert("RGB")
                
                # Reduz para no máximo 400x400px (preserva proporção)
                # Isso deixa a busca MUITO mais rápida do que enviar a imagem original
                img.thumbnail((400, 400)) 
                
                # Salva em memória (buffer)
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=70)
                
                # Converte para string Base64
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                img_base64 = f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            print(f"Erro ao converter imagem {file_path}: {e}")
            img_base64 = None # Retorna sem imagem se der erro

        results.append({
            'path': file_path, 
            'similarity': max(0.0, similarity),
            'image': img_base64  # Novo campo com a imagem pronta para HTML
        })
        
    return results

# =========================================================================
# 4. ROTAS DO FLASK
# =========================================================================

@app.route('/')
def home():
    """Rota inicial que renderiza a interface de busca."""
    # Envia o status do indice para o frontend
    status = 'OK' if index is not None and index.ntotal > 0 else 'PENDING'
    total_items = index.ntotal if index is not None else 0
    return render_template('index.html', status=status, total_items=total_items)

@app.route('/indexar', methods=['POST'])
def index_data():
    """Endpoint para iniciar a indexação das imagens."""
    global index, mapping
    
    # Não permitir reindexação se o índice já existe
    if index is not None:
        return jsonify({'message': f'Indexacao ja concluida ({index.ntotal} itens). Reinicie para reindexar se houver mudancas.', 'success': True})

    try:
        image_paths = get_image_paths(IMAGE_ROOT_DIR)
        if not image_paths:
            return jsonify({'message': f'Erro: Nenhuma imagem encontrada no diretorio configurado: {IMAGE_ROOT_DIR}', 'success': False})

        # Processamento e Indexação (as etapas mais lentas)
        embeddings, processed_paths = generate_embeddings(image_paths, model, processor, device)
        index, mapping = create_faiss_index(embeddings, processed_paths)
        
        if index is None:
             return jsonify({'message': 'Falha na geracao de embeddings ou na criacao do indice.', 'success': False})

        return jsonify({'message': f'Indexacao concluida com sucesso! Total de {index.ntotal} itens indexados.', 'success': True, 'total_items': index.ntotal})

    except Exception as e:
        print(f"ERRO DE INDEXACAO: {e}")
        return jsonify({'message': f'Erro fatal durante a indexacao: {e}', 'success': False})


@app.route('/search', methods=['POST'])
def search():
    """Endpoint para realizar a busca semântica."""
    if index is None:
        return jsonify({'message': 'O indice de busca ainda nao foi criado. Por favor, execute a indexacao primeiro.', 'success': False}), 400

    try:
        data = request.json
        
        # O frontend envia query_type e query_value
        query_vector = get_vector_from_input(data, model, processor, device)
        
        # Busca
        results = search_index(query_vector, index, mapping, k=20)
        
        # O jsonify usa o NpEncoder customizado, resolvendo o problema de serialização.
        return jsonify({'results': results, 'success': True})

    except Exception as e:
        print(f"ERRO DE BUSCA: {e}")
        return jsonify({'message': f'Erro ao processar a busca: {e}', 'success': False}), 500

@app.route('/open-folder', methods=['POST'])
def open_local_folder():
    """Abre a pasta no Windows Explorer e SELECIONA o arquivo."""
    try:
        data = request.json
        file_path = data.get('path')
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'success': False, 'message': 'Caminho não encontrado.'})

        # Normaliza o caminho para o formato do Windows (barras invertidas)
        file_path = os.path.normpath(file_path)
        
        # O comando "explorer /select,C:\Caminho\Arquivo.jpg" abre a pasta e foca no arquivo
        subprocess.Popen(f'explorer /select,"{file_path}"')
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Erro ao abrir pasta: {e}")
        return jsonify({'success': False, 'message': str(e)})

# =========================================================================
# 5. INICIALIZAÇÃO DO APP E EXECUÇÃO
# =========================================================================

if __name__ == '__main__':
    
    # 1. CARREGAR MODELO: Executa o carregamento da IA uma vez ao iniciar o app
    if not load_model():
        print("AVISO: O aplicativo será executado, mas a busca não funcionará sem o modelo CLIP.")
    
    # 2. CARREGAR ÍNDICE: Tenta carregar o índice FAISS e o mapeamento
    load_index_and_mapping() 
    
    # 3. INICIAR FLASK
    print("\n" + "="*50)
    print(" FLASK PRONTO. ACESSE: http://127.0.0.1:5000/ ")
    print("="*50)

    # Nota: Mantenho debug=False pois a indexação já foi concluída, mas você pode mudar para debug=True para desenvolvimento.
    app.run(debug=False, use_reloader=False, port=5000)
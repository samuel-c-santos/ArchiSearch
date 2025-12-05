# ArchiSearch: Semantic Search for Architectural References

## Contexto e Motivação

Minha trajetória profissional iniciou-se como Técnico em Edificações, atuando na modelagem de projetos arquitetônicos e suporte a engenharia civil. Ao longo dos anos, acumulei uma biblioteca pessoal de referências visuais com milhares de arquivos.

Contudo, a organização tradicional de arquivos em pastas revelou-se ineficiente para a natureza fluida da arquitetura. Uma única imagem de uma fachada pode conter elementos que se enquadram simultaneamente em "Casas de Campo", "Estilo Colonial", "Uso de Madeira" e "Jardins Verticais". Categorizá-la em uma única pasta limitava sua descoberta futura.

Frequentemente, ao iniciar um novo projeto, eu enfrentava um paradoxo: possuía a referência exata armazenada localmente, mas a dificuldade de encontrá-la era tamanha que se tornava mais rápido realizar uma nova busca genérica na web. O ativo intelectual que eu havia curado estava inacessível devido à rigidez do sistema de arquivos.

Unindo esta demanda à minha atuação atual em Análise de Dados, Geoprocessamento e Desenvolvimento de Sistemas, desenvolvi o **ArchiSearch**. Este projeto aplica inteligência artificial para romper a barreira da busca por nome de arquivo, permitindo localizar referências locais através de contexto semântico e similaridade visual.

## Como Funciona

O sistema não utiliza tags manuais. Ele utiliza o modelo **CLIP (Contrastive Language-Image Pre-training)** da OpenAI para "ler" as imagens e convertê-las em vetores matemáticos (embeddings).

1.  **Indexação:** O sistema varre o diretório local de imagens e gera um vetor multidimensional para cada arquivo.
2.  **Armazenamento Vetorial:** Utilizamos o **FAISS (Facebook AI Similarity Search)** para indexar esses vetores, permitindo buscas de altíssima performance.
3.  **Busca Semântica:** Ao digitar "fachada de vidro moderna", o sistema converte o texto em um vetor e busca as imagens cujos vetores estejam matematicamente próximos, independentemente do nome do arquivo.
4.  **Busca Visual:** O usuário pode fazer upload de uma imagem de referência e o sistema localizará arquivos visualmente similares no acervo local.

### A Lógica por Trás (Deep Dive)

Tradicionalmente, computadores não "veem" imagens como nós; eles veem grades de pixels. Para buscar uma "casa colonial" num sistema comum, você precisaria ter renomeado o arquivo manualmente. O ArchiSearch elimina essa necessidade.

Imagine que o sistema traduz tanto as **Imagens** quanto os **Textos** para uma língua universal: a matemática.

* **O Tradutor (Encoder):** Quando o sistema indexa sua pasta, ele passa cada imagem por uma rede neural que extrai características visuais (formas, texturas, estilos) e as converte em uma lista de números (um vetor).
* **O Mapa (Espaço Latente):** Esses vetores são plotados em um espaço multidimensional. Neste espaço, uma foto de uma "Cabana de Madeira" fica matematicamente muito próxima do vetor da palavra "Rústico" ou "Madeira", mesmo que essas palavras nunca tenham sido escritas no arquivo.
* **A Bússola (FAISS):** Quando você busca, o sistema calcula a distância matemática entre o que você pediu e o que existe no disco. O resultado é pura similaridade semântica.

```mermaid
flowchart LR
    subgraph Indexing ["1. Indexação (Backend)"]
        direction TB
        Img[("📂 Arquivo de Imagem<br/>'casa_01.jpg'")] -->|Leitura| EncImg[("🧠 CLIP Image Encoder")]
        EncImg -->|Transformação| VecImg["🔢 Vetor Matemático<br/>(Embedding)"]
        VecImg --> DB[("🗄️ Banco de Vetores<br/>(FAISS Index)")]
    end

    subgraph Searching ["2. Busca (Usuário)"]
        direction TB
        Query["👤 Texto: 'Fachada Colonial'<br/>OU<br/>🖼️ Imagem de Referência"] -->|Input| EncTxt[("🧠 CLIP Encoder<br/>(Texto ou Imagem)")]
        EncTxt -->|Transformação| VecQuery["🔢 Vetor de Busca"]
    end

    VecQuery -- "3. Cálculo de Distância (Similaridade)" --> DB
    DB --> Result["✅ Resultados:<br/>Imagens com vetores<br/>matematicamente próximos"]

    style Indexing fill:#f0fdf4,stroke:#15803d,stroke-width:2px
    style Searching fill:#eff6ff,stroke:#1d4ed8,stroke-width:2px
    style Result fill:#fff7ed,stroke:#c2410c,stroke-width:2px,stroke-dasharray: 5 5
````

## Demonstração

### 1. Interface Inicial

O sistema apresenta um dashboard limpo, indicando o status da indexação e oferecendo as duas modalidades de busca.

![Tela Inicial](.\static\demo\inicial.png)

### 2. Busca Textual (Contexto)

Exemplo de busca por termos específicos como "Estilo Colonial". O modelo compreende as características arquitetônicas (telhados, colunas, cores) sem que a palavra "colonial" precise estar no nome do arquivo.

![Busca por Texto](.\static\demo\text_colonial.png)

### 3. Busca por Similaridade de Imagem

Exemplo utilizando uma imagem de referência de uma "Cabana". O algoritmo identifica padrões de forma, textura e composição para retornar projetos similares do acervo.

![Busca por Imagem](.\static\demo\img_cabana.png)

## Funcionalidades

  * **Busca por Linguagem Natural:** Consultas complexas em português ou inglês (ex: "interiores com iluminação natural").
  * **Reverse Image Search:** Upload de imagem para encontrar similares no disco local.
  * **Deep Link com o SO:** Botão dedicado para abrir a pasta do arquivo diretamente no Windows Explorer, com o arquivo selecionado.
  * **Performance:** Uso de indexação FAISS para respostas instantâneas mesmo em grandes volumes de dados.
  * **Privacidade:** Todo o processamento é local (On-Premise), sem envio de imagens para nuvens de terceiros.

## Stack Tecnológico

  * **Linguagem:** Python 3.x
  * **Core AI:** PyTorch, Transformers (Hugging Face), OpenAI CLIP.
  * **Indexação Vetorial:** FAISS (Facebook AI Similarity Search).
  * **Backend/API:** Flask.
  * **Frontend:** HTML5, JavaScript (Vanilla), TailwindCSS.

## Instalação e Execução

### Pré-requisitos

  * Python 3.8 ou superior.
  * Placa de vídeo com suporte a CUDA (recomendado para indexação rápida, mas funciona em CPU).

### Passos

1.  Clone o repositório:

    ```bash
    git clone [https://github.com/samuel-c-santos/archisearch.git](https://github.com/samuel-c-santos/archisearch.git)
    cd archisearch
    ```

2.  Instale as dependências:

    ```bash
    pip install -r requirements.txt
    ```

3.  Configuração:
    Abra o arquivo `app.py` e edite a variável `IMAGE_ROOT_DIR` para apontar para sua pasta de referências:

    ```python
    IMAGE_ROOT_DIR = r"C:\Caminho\Para\Suas\Referencias"
    ```

4.  Execute a aplicação:

    ```bash
    python app.py
    ```

5.  Acesse no navegador:
    `http://127.0.0.1:5000`

-----

## Autor

**Samuel Santos** *Geoprocessamento | Data Science | Edificações.*

[](https://samuel-c-santos.github.io/)
[](https://www.linkedin.com/in/samuelsantos-amb/)
[](https://github.com/samuel-c-santos)

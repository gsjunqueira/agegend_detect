# Projeto de Detecção de Idade e Gênero

Este projeto implementa um sistema de detecção de rostos em tempo real, seguido da predição de idade e gênero utilizando redes neurais convolucionais pré-treinadas. Ele utiliza **OpenCV** para processamento de imagens e vídeo, e **Caffe** para carregar os modelos de detecção de rosto, classificação de idade e classificação de gênero.

## Funcionalidades

- **Detecção de Rosto**: Detecta rostos em tempo real a partir da webcam.
- **Classificação de Gênero**: Classifica o gênero em Masculino ou Feminino para cada rosto detectado.
- **Estimativa de Idade**: Estima a faixa etária aproximada (ex. `(25, 32)`, `(38, 43)`) para cada rosto detectado.
- **Visualização em Tempo Real**: Exibe os resultados de idade e gênero diretamente no vídeo ao vivo.

## Tecnologias Utilizadas

- **Python 3.x**
- **OpenCV 4.x**
- **Caffe**
- **Numpy**
- **Poetry** para gerenciamento de dependências

## Modelos Utilizados

Este projeto faz uso de três modelos pré-treinados fornecidos pelo Caffe:

- **Detecção de Rostos**: `res10_300x300_ssd_iter_140000_fp16.caffemodel`
- **Classificação de Gênero**: `deploy_gender.prototxt` e `gender_net.caffemodel`
- **Classificação de Idade**: `deploy_age.prototxt` e `age_net.caffemodel`

## Instalação

### Usando o Poetry

1. Clone o repositório para sua máquina local:

    ```bash
    git clone https://github.com/seu-usuario/projeto-deteccao-idade-genero.git
    ```

2. Instale o Poetry, caso ainda não tenha instalado:

    ```bash
    pip install poetry
    ```

3. Instale as dependências do projeto com o Poetry:

    ```bash
    poetry install
    ```

4. Baixe os arquivos de modelo Caffe e coloque-os no diretório correto (`agegend_detect/set/`):
   - `deploy_gender.prototxt`
   - `gender_net.caffemodel`
   - `deploy_age.prototxt`
   - `age_net.caffemodel`
   - `res10_300x300_ssd_iter_140000_fp16.caffemodel`

5. Ative o ambiente virtual do Poetry:

    ```bash
    poetry shell
    ```

## Como Executar

1. Certifique-se de que os arquivos de modelo estão no caminho correto (diretório `agegend_detect/set/`).

2. Execute o script principal:

    ```bash
    poetry run python nome_do_arquivo.py
    ```

3. A webcam será iniciada e o sistema irá detectar rostos em tempo real. Para cada rosto, o gênero e a faixa etária estimada serão exibidos na tela.

4. Pressione `q` para sair da visualização.

## Estrutura do Projeto

├── agegend_detect/
│   └── set/
│       ├── deploy_gender.prototxt
│       ├── gender_net.caffemodel
│       ├── deploy_age.prototxt
│       ├── age_net.caffemodel
│       ├── res10_300x300_ssd_iter_140000_fp16.caffemodel
├── main.py
├── pyproject.toml
├── README.md

## Exemplos de Saída

Quando um rosto for detectado, a idade e o gênero estimados serão exibidos diretamente na imagem:

- **Exemplo de saída**:

Masculino - 98.3%, (25, 32) - 85.7%

A caixa em volta do rosto será colorida de **azul** se for **Masculino** e **rosa** se for **Feminino**.

## Dependências

- Python 3.x
- OpenCV 4.x
- Numpy
- Modelos pré-treinados Caffe (detecção de rostos, classificação de idade e gênero)
- Poetry

## Contribuindo

Se você quiser contribuir com este projeto, sinta-se à vontade para fazer um **fork**, criar uma **branch** com sua funcionalidade (git checkout -b funcionalidade/MinhaFuncionalidade), e depois enviar um **Pull Request**.

## Licença

Este projeto é licenciado sob a [MIT License](LICENSE).

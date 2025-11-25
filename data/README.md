# 📥 Instruções para o Dataset (Base ORL Faces)

Este projeto utiliza a base de dados clássica **ORL Faces** (Olivetti Research Laboratory) para o reconhecimento facial. O dataset contém 400 imagens, 10 para cada um dos 40 sujeitos.

**ATENÇÃO:** O dataset completo não está incluído neste repositório do Git devido ao seu tamanho e por ser uma fonte externa.

## 1. Localização do Dataset

Para rodar o script `src/reconhecimento_facial.py`, você deve:

1.  Baixar o dataset **ORL Faces** ([DataBase](https://pucdegoias-my.sharepoint.com/:f:/g/personal/20221003300956_pucgo_edu_br/IgD9MmixodluRZTf3LGdAshRAXdelcqhV2fcbzbQ1odUf5Y?e=2sugKE)).
2.  Descompactar a pasta.
3.  Colocar a pasta resultante (`orl_faces`) diretamente na **raiz** do projeto (ao lado da pasta `data/` e `src/`).

**O script espera encontrar o dataset neste caminho:**
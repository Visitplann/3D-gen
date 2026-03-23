# Como Correr

Este projeto lê sempre as 5 imagens de `input/test_obj` e gera um modelo 3D simples em `python/output`.

## O que tens de ter

- `Python 3.11`
- ambiente virtual `.venv`
- dependências instaladas com `pip install -r requirements.txt`

## Estrutura esperada do input

A pasta `input/test_obj` tem de ter estes ficheiros:

- `frente.jpg`
- `traz.jpg`
- `esquerda.jpg`
- `direita.jpg`
- `cima.jpg`

Se faltar um destes ficheiros, o pipeline falha com erro claro.

## Como correr

Na raiz do projeto:

```bash
cd 3D-gen
PYTHONPATH=python MPLBACKEND=Agg ./.venv/bin/python python/pipeline.py
```

## Onde ficam os outputs

Depois de correr, os ficheiros ficam aqui:

- modelo final: `python/output/model/test_obj.glb`
- material simples: `python/output/materials/albedo.png`
- report da execução: `python/output/reports/test_obj_report.txt`
- imagens de debug: `python/output/debug/test_obj/`

## Como ler o debug

Dentro de `python/output/debug/test_obj/`:

- `views/` tem as máscaras e overlays de cada vista
- `reconstruction/` tem as projeções usadas para montar o volume

## Comando útil para limpar outputs antigos

Se quiseres apagar os outputs antes de correr outra vez:

```bash
rm -rf python/output/model python/output/materials python/output/reports python/output/debug
```

## Resultado esperado

O pipeline tenta gerar um objeto simples e fiel à forma exterior principal.
Não tenta modelar detalhes pequenos como portas, ranhuras finas ou logótipos.

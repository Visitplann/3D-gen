# Como Correr

Este projeto lê sempre as 5 imagens de `input/test_obj` e gera um modelo 3D simples em `python/output`.

## O que precisas

- `Python 3.11`
- `git`
- um ambiente virtual `.venv`

## Como preparar noutro PC

Na primeira vez:

```bash
git clone <url-do-repo>
cd 3D-gen
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input esperado

A pasta `input/test_obj` tem de ter exatamente estes ficheiros:

- `frente.jpg`
- `traz.jpg`
- `esquerda.jpg`
- `direita.jpg`
- `cima.jpg`

Se faltar um destes ficheiros, o pipeline falha.

## Como correr no terminal

Na raiz do projeto:

```bash
cd 3D-gen
PYTHONPATH=python MPLBACKEND=Agg ./.venv/bin/python python/pipeline.py
```

## Como correr no VS Code

1. Abre a pasta `3D-gen` no VS Code.
2. Se o VS Code pedir para escolher Python, escolhe `.venv/bin/python`.
3. Vai a `Run and Debug`.
4. Escolhe `Run 5-View Pipeline`.
5. Carrega em `F5` ou no botão `Run`.

O projeto já traz estes ficheiros:

- `.vscode/settings.json`
- `.vscode/launch.json`

Esses ficheiros fazem o VS Code usar:

- o Python da `.venv`
- `PYTHONPATH=python`
- `MPLBACKEND=Agg`

Importante:

- não uses `Python Debugger: Current File`
- não tentes correr `.vscode/launch.json`
- `launch.json` é só configuração do VS Code, não é um script Python

## Onde ficam os outputs

Depois de correr, o output fica organizado assim:

- modelo final: `python/output/model/test_obj.glb`
- material simples: `python/output/materials/albedo.png`
- report: `python/output/reports/test_obj_report.txt`
- debug: `python/output/debug/test_obj/`

## Como ler o debug

Dentro de `python/output/debug/test_obj/`:

- `views/` tem a máscara e o overlay de cada foto
- `reconstruction/` tem as projeções usadas para montar o volume

## Como limpar outputs antigos

Se quiseres apagar os outputs antes de correr outra vez:

```bash
rm -rf python/output/model python/output/materials python/output/reports python/output/debug
```

## O que esperar do resultado

O pipeline tenta gerar a forma exterior principal do objeto.
Não tenta modelar detalhes pequenos como portas, ranhuras finas ou logótipos.

É normal aparecerem warnings no terminal.
Esses warnings não significam sempre erro.
Servem para avisar quando uma vista está mais fraca, por exemplo a `frente`.

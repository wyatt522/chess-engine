# chess-engine

Chess engine based on the Maia-1 model with finetuning capabilities. To finetune a model, edit the `training_config.yaml` file to the desired chess.com username, and run `mix_data_finetuning`. `generate_dataset.py` can create a dataset given the Chess.com username in the configuration.

## Setup:

- Install Python dependencies:

    ```pip install -r requirements.txt```

- Put your data (.pgn files) into ```data/pgn/```. 



> The [dataset](https://database.nikonoel.fr/) that I used.



Install tablebase: https://chess.cygnitec.com/tablebases/gaviota/

## Installing UCI:

- Quick Folder install:

    ```pyinstaller --onedir --distpath ./dist --name MiniMaiaBot --add-data "uci_config.yaml:." engines/torch/uci.py```
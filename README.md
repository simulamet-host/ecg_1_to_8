# ECG_1_to_8

## [Paper](TBD) | [GitHub](https://github.com/simulamet-host/ecg_1_to_8) | [Trained Model 1 lead input](https://drive.google.com/file/d/1wdUp-Rs2TIjKDw1ih5NaTNUtlFyDxTYC/view) | [Trained Model 2 leads input](https://drive.google.com/file/d/1jExuEZxfo7sy_R6EPcFLbEu3FG7KYOe7/view) | [Example ECGs from the PTB dataset](https://drive.google.com/file/d/11yZi6Oxtt2A-AHmhJYhlGJ2zFJVFm2uM/view)
---

Generate ECG leads from a single-lead or dual-lead as input using deep learning algorithms.


## Installation

Clone the repository and install the requirements 

```bash
pip install -r requirements.txt
```

To prepare the PTB-XL dataset, ensure the `PATH_TO_PTB_DATA` value from `convert_ptb.py` is correct, then run the following command:
    
```bash
python convert_ptb.py
```

## Usage examples

Training:

One lead as input
```bash
    python main.py --action=train --dataset=PTB --network=GAN --model-size=32 --epochs=200 --save-every=5 --input-leads=1
```
Two leads as input
```bash
    python main.py --action=train --dataset=PTB --network=GAN --model-size=32 --epochs=200 --save-every=5 --input-leads=2
```

Generate outputs from the test datasets using a trained model:
```bash
    python main.py --action=generate_outputs --dataset=PTB --network=GAN --model-size=32 --saved-model=./test_models/gan_1lead_checkpoint.pt --outputs-folder=test_models --plots --csv --limit=100
```

```bash
    python main.py --action=generate_outputs --dataset=PTB --network=GAN --model-size=32 --saved-model=./test_models/gan_2leads_checkpoint.pt --outputs-folder=test_models --plots --csv --limit=100 --input-leads=2
```

```bash
    python main.py --action=generate_outputs --dataset=PTB --network=GAN --model-size=32 --saved-model=./test_models/gan_1lead_checkpoint.pt --plots
```

```bash
    python main.py --action=generate_outputs --dataset=PTB --network=GAN --model-size=32 --saved-model=./test_models/gan_1lead_checkpoint.pt --plots --seconds=2.5 --columns=4 --limit=5
```

```bash
    python main.py --action=generate_outputs --dataset=PTB --network=GAN --model-size=32 --saved-model=./test_models/gan_1lead_checkpoint.pt --plots --seconds=2.5 --columns=4 --range=5.8 --index=8569 --index=8616
```

Generate outputs from a csv:
```bash
    python main.py --action=generate --input=./test_models/example_input.csv --saved-model=./test_models/gan_1lead_checkpoint.pt --outputs-folder=test_models --plots --csv
```

```bash
    python main.py --action=generate --input=./test_models/example_input.csv --network=GAN --model-size=32 --saved-model=./test_models/gan_1lead_checkpoint.pt --outputs-folder=test_models --plots --seconds=10 --columns=4 --range=1.8 --csv --normalize-factor=8
```

```bash
    python main.py --action=generate --input=./test_models/example_input_2leads.csv --saved-model=./test_models/gan_2leads_checkpoint.pt  --input-leads=2 --outputs-folder=test_models --plots --csv
```

Testing (compute metrics on the test dataset):
```bash
    python main.py --action=test --dataset=PTB --network=GAN --model-size=32 --saved-model=./test_models/gan_1lead_checkpoint.pt
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Citation:
```latex
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

## For more details:
Please contact: alexandru.dorobantiu@ulbsibiu.ro, vajira@simula.no

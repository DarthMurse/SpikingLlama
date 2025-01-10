# Auto Quantization Encoder
This repository aims to use a fully connected network and a QCFS neuron to implement a 1-step quantization for all the input vectors before a linear layer in a GPT model. The feasibilty of this method is still under investigation. Our network architecture, training spefications and codes are based on [TinyLlama](https://github.com/jzhang38/TinyLlama).

## Preparation
To run the script, please first configure the python environment using the following scipts.

	conda create -n SpikingLlama python==3.11
	pip3 install -r requirements

We also need the rotatory embedding and the fused entropy script from flash attention. To install the module, run the following scripts.

	git clone https://github.com/Dao-AILab/flash-attention
	cd flash-attention
	cd csrc/rotary && pip install .
	cd ../xentropy && pip install .

## Datasets
We use the openwebtext dataset from huggingface for pretraining. To download the dataset, please use the following script.

	cd data
	export HF_ENDPOINT=https://hf-mirror.com
	huggingface-cli download --repo-type dataset --resume-download Skylion007/openwebtext --local-dir openwebtext

After downloading the dataset, please use the following script to process the dataset.

	python3 -m scripts.prepare_openwebtext

It can take some time to preprocess the full dataset. After processing, there should be a directory named `openwebtext_preprocessed` under the `data` directory.

## Training
After preprocesing the dataset, you need to first change the code in the file `src/quant_model.py`. Specifically, uncomment line 290 and comment line 291.

	return build_rope_cache(
		...,
		dtype=torch.bfloat16,
		#dtype=idx.dtype,
		...
		)

Then simply using the following script.

	./pretrain.sh
	# This is default to multi-card training, which uses 4 cards of an A100 or H100 server.
	# To change the number of cards, modify the script `pretrain.sh` and the GPU_NUM parameter in `pretrain.py`.

The output checkpoints are stored in the `out` directory. 

For details of training and methods, you can refer to the codes in `pretrain.py` and `src/quant_model.py`.

## Evaluation
To evaluate the model, we use the lm-evaluation-harness toolset. To install lm-eval, use the following scripts.

	git clone https://github.com/EleutherAI/lm-evaluation-harness
	cd lm-evaluation-harness
	pip install -e .

Then you can use `eval.py`, but be aware to change the model and paths in the python scripts to make sure you are evaluating the model you want. (If you use the model trained by yourself, the checkpoint is stored under the directory `out/spike-llama-120M`) Before running the following script, you need to change the code of `src/quant_model.py` by commenting line 290 and uncommenting line 291.

	return build_rope_cache(
		...,
		#dtype=torch.bfloat16,
		dtype=idx.dtype,
		...
		)

Then you can use the following script.

	export HF_ENDPOINT=https://hf-mirror.com    # huggingface is inaccessible in China.
	./eval.sh

metrics_path = "metrics"
datasets = ["bank-marketing","jannis",
            "electricity","MiniBooNE","phoneme"]
models = ["rf","linear"]
decompositions = ["pca","fa","fastica","ipca","ae","vime","none"]
fractions = [0.0,0.05,0.1,0.5,0.75,1.0]
outputs = []
for dataset in datasets:
    for model in models:
        for decomposition in decompositions:
            for fraction in fractions:
                output = "{}/{}_{}_{}_{}.json".format(
                    metrics_path,dataset,model,decomposition,fraction)
                outputs.append(output)
            
rule all:
    input:
        outputs

rule train:
    input:
        "src/__main__.py"
    output:
        "metrics/{dataset}_{model}_{decomposition}_{fraction}.json"
    shell:
        """
        mkdir -p metrics

        python -m src \
            --dataset {wildcards.dataset} \
            --decomposition {wildcards.decomposition} \
            --learning_algorithm {wildcards.model} \
            --unsupervised_fraction {wildcards.fraction} \
            --seed 42 \
            --n_folds 10 \
            --output_file {output}
        """
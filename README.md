# BERT-SNER
Repository for German clinical information extraction with BERT-SNER.

# Checkpoints Host
https://huggingface.co/sitingGZ/german-bert-clinical-ner/tree/main

# Use
    import sys
    sys.path.append('modules')
    
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM, EncoderDecoderConfig
    from BERT2span_semantic_disam import BERT2span
    from helpers import load_config, set_seed
    from inference import final_label_results_rescaled

    base_name =  "bert-base-german-cased"
    configs = load_config('configs/step3_gpu_span_semantic_group.yaml')
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    bertMLM = AutoModelForMaskedLM.from_pretrained(base_name)
    bert_sner = BERT2span(configs, bertMLM, tokenizer)
    
    checkpoint_path = "checkpoints/german_bert_ex4cds_500_semantic_term.ckpt"
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    bert_sner.load_state_dict(state_dict)
    bert_sner.eval()
    
    suggested_terms = {'Condition': 'Zeichen oder Symptom',
                       'DiagLab': 'Diagnostisch und Laborverfahren',
                        'LabValues': 'Klinisches Attribut',
                         'HealthState': 'Gesunder Zustand',
                         'Measure': 'Quantitatives Konzept',
                         'Medication': 'Pharmakologische Substanz',
                         'Process': 'Physiologische Funktion',
                         'TimeInfo': 'Zeitliches Konzept'}

    words = "Aktuell Infekt mit Nachweis von E Coli und Pseudomonas im TBS- CRP 99mg/dl".split()
    words_list = [words]
    heatmaps, ner_results = final_label_results_rescaled(words_list, tokenizer, bert_sner, suggested_terms, threshold=0.5)
    
    
    

    
    
    

# Citation
    @inproceedings{liang-etal-2023-cross,
    title = "Cross-domain {G}erman Medical Named Entity Recognition using a Pre-Trained Language Model and Unified Medical Semantic Types",
    author = "Liang, Siting  and
      Hartmann, Mareike  and
      Sonntag, Daniel",
    booktitle = "Proceedings of the 5th Clinical Natural Language Processing Workshop",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.clinicalnlp-1.31",
    doi = "10.18653/v1/2023.clinicalnlp-1.31",
    pages = "259--271",}

# Task Datasets

# i) Amazon Reviews Dataset
amazon_reviews_dataset = [
    {
      "review_id": [0],
      "label": [0],
      "title": ["Completely useless"],
      "text": ["The cable didn’t work at all. My phone never recognized it, and it felt flimsy right out of the box. Total waste of money."]
    },
    {
      "review_id": [0],
      "label": [1],
      "title": ["Below expectations"],
      "text": ["The blender works, but it struggles with anything harder than a banana. Very loud and the plastic feels cheap. I expected better for the price."]
    },
    {
      "review_id": [0],
      "label": [2],
      "title": ["Just okay"],
      "text": ["The keyboard is fine for basic typing, but the keys are a bit stiff and the backlight is uneven. Not terrible, but nothing special either."]
    },
    {
      "review_id": [0],
      "label": [3],
      "title": ["Pretty good overall"],
      "text": ["The vacuum has strong suction and is easy to maneuver. The dust container is a bit small, but for the price it’s a solid choice."]
    },
    {
      "review_id": [0],
      "label": [4],
      "title": ["Exceeded my expectations"],
      "text": ["Amazing sound quality for such a small speaker. Battery life is great, Bluetooth connects instantly, and it feels very well made. I’d buy it again."]
    },
]

# ii) WOS Dataset
WOS_dataset = [
    {
        "text": ["""Phytoplasmas are insect-vectored bacteria that cause disease in a wide range of plant species. The increasing availability of molecular DNA analyses, expertise and additional methods in recent years has led to a proliferation of discoveries of phytoplasma-plant host associations and in the numbers of taxonomic groupings for phytoplasmas. The widespread use of common names based on the diseases with which they are associated, as well as separate phenetic and taxonomic systems for classifying phytoplasmas based on variation at the 16S rRNA-encoding gene, complicates interpretation of the literature. We explore this issue and related trends through a focus on Australian pathosystems, providing the first comprehensive compilation of information for this continent, covering the phytoplasmas, host plants, vectors and diseases. Of the 33 16Sr groups reported internationally, only groups I, II, III, X, XI and XII have been recorded in Australia and this highlights the need for ongoing biosecurity measures to prevent the introduction of additional pathogen groups. Many of the phytoplasmas reported in Australia have not been sufficiently well studied to assign them to 16Sr groups so it is likely that unrecognized groups and sub-groups are present. Wide host plant ranges are apparent among well studied phytoplasmas, with multiple crop and non-crop species infected by some. Disease management is further complicated by the fact that putative vectors have been identified for few phytoplasmas, especially in Australia. Despite rapid progress in recent years using molecular approaches, phytoplasmas remain the least well studied group of plant pathogens, making them a "crouching tiger" disease threat."""],
        "coarse_label": [2],
        "fine_label": [9]
    },
    {
        "text": ["""Background: (-)-alpha-Bisabolol, also known as levomenol, is an unsaturated sesquiterpene alcohol that has mainly been used in pharmaceutical and cosmetic products due to its anti-inflammatory and skin-soothing properties. (-)-alpha-Bisabolol is currently manufactured mainly by steam-distillation of the essential oils extracted from the Brazilian candeia tree that is under threat because its natural habitat is constantly shrinking. Therefore, microbial production of (-)-alpha-bisabolol plays a key role in the development of its sustainable production from renewable feedstock. Results: Here, we created an Escherichia coli strain producing (-)-alpha-bisabolol at high titer and developed an in situ extraction method of (-)-alpha-bisabolol, using natural vegetable oils. We expressed a recently identified (-)-alpha-bisabolol synthase isolated from German chamomile (Matricaria recutita) (titer: 3 mg/L), converted the acetyl-CoA to mevalonate, using the biosynthetic mevalonate pathway (12.8 mg/L), and overexpressed farnesyl diphosphate synthase to efficiently supply the (-)-alpha-bisabolol precursor farnesyl diphosphate. Combinatorial expression of the exogenous mevalonate pathway and farnesyl diphosphate synthase enabled a dramatic increase in (-)-alpha-bisabolol production in the shake flask culture (80 mg/L) and 5 L bioreactor culture (342 mg/L) of engineered E. coli harboring (-)-alpha-bisabolol synthase. Fed-batch fermentation using a 50 L fermenter was conducted after optimizing culture conditions, resulting in efficient (-)-alpha-bisabolol production with a titer of 9.1 g/L. Moreover, a green, downstream extraction process using vegetable oils was developed for in situ extraction of (-)-alpha-bisabolol during fermentation and showed high yield recovery (>98%). Conclusions: The engineered E. coli strains and economically viable extraction process developed in this study will serve as promising platforms for further development of microbial production of (-)-alpha-bisabolol at large scale."""],
        "coarse_label": [2],
        "fine_label": [7]
    },
    {
        "text": ["""A universal feature of the replication of positive-strand RNA viruses is the association with intracellular membranes. Carnation Italian ringspot virus (CIRV) replication in plants occurs in vesicles derived from the mitochondrial outer membrane. The product encoded by CIRV ORF1, p36, is required for targeting the virus replication complex to the outer mitochondrial membrane both in plant and yeast cells. Here the yeast Saccharomyces cerevisiae was used as a model host to study the effect of CIRV p36 on cell survival and death. It was shown that p36 does not promote cell death, but decreases cell growth rate. In addition, p36 changed the nature of acetic acid-induced cell death in yeast by increasing the number of cells dying by necrosis with concomitant decrease of the number of cells dying by programmed cell death, as judged by measurements of phosphatidylserine externalization. The tight association of p36 to membranes was not affected by acetic acid treatment, thus confirming the peculiar and independent interaction of CIRV p36 with mitochondria in yeast. This work proved yeast as an invaluable model organism to study both the mitochondrial determinants of the type of cell death in response to stress and the molecular pathogenesis of (+)RNA viruses. (C) 2016 Elsevier Ireland Ltd. All rights reserved."""],
        "coarse_label": [2],
        "fine_label": [7]
    },
    {
        "text": ["""1,2-Dichloropropane (1,2-DCP) and dichloromethane (DCM) are possible causative agents associated with the development of cholangiocarcinoma in employees working in printing plant in Osaka, Japan. However, few reports have demonstrated an association between these agents and cholangiocarcinoma in rodent carcinogenicity studies. Moreover, the combined effects of these compounds have not been fully elucidated. In the present study, we evaluated the in vivo mutagenicity of 1,2-DCP and DCM, alone or combined, in the livers of gpt delta rats. Six-week-old male F344 gpt delta rats were treated with 1,2-DCP, DCM or 1,2-DCP+DCM by oral administration for 4weeks at the dose (200mgkg(-1) body weight 1,2-DCP and 500mgkg(-1) body weight DCM) used in the carcinogenesis study performed by the National Toxicology Program. In vivo mutagenicity was analyzed by gpt mutation/Spi(-) assays in the livers of rats. In addition, gene and protein expression of CYP2E1 and GSTT1, the major enzymes responsible for the genotoxic effects of 1,2-DCP and DCM, were analyzed by quantitative polymerase chain reaction and western blotting. Gpt and Spi(-) mutation frequencies were not increased by 1,2-DCP and/or DCM in any group. Additionally, there were no significant changes in the gene and protein expression of CYP2E1 and GSTT1 in any group. These results indicated that 1,2-DCP, DCM and 1,2-DCP+DCM had no significant impact on mutagenicity in the livers of gpt delta rats under our experimental conditions. Copyright (c) 2016 John Wiley & Sons, Ltd."""],
        "coarse_label": [2],
        "fine_label": [9]
    },
    {
        "text": ["""This paper presents the simulation results of a linear, fully integrated, two-stage digitally programmable 130 nm CMOS power amplifier (PA) operating at 2.4 GHz. Its power stage is composed of a set of amplifying cells which can be enabled or disabled independently by a digital control circuit. All seven operational modes are univocal in terms of 1 dB output compression point (OCP1dB), saturated output power (P-SAT) and power gain at 2.4 GHz. The lowest power mode achieves an 8.1 dBm P-SAT, a 13.5 dB power gain and consumes 171 mW DC power (P-DC) at an OCP1dB of 6 dBm, whereas the highest power mode reaches an 18.9 dBm P-SAT and a 21.1 dB power gain and consumes 415 mW P-DC at an OCP1dB of 18.2 dBm."""],
        "coarse_label": [0],
        "fine_label": [1]
    },
]

# iii) Entity Extraction with IOB Tags (CoNLL)
# label meanings: 2: B, 0: I, 1: O
IOB_dataset = [
    {
        "tokens" : ["Sydney", "Airport", "announces", "range", "of", "new", "sustainability", "goals", "Finland", "is", "on", "track", "to", "be", "carbon", "neutral", "by", "2035"],
        "labels": [2, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 0, 1, 2],
        "labels_tags" : ["B", "I", "O", "O", "O", "O", "O", "O", "B", "O", "O", "O", "O", "O", "B", "I", "O", "B"]
    },
    {
        "tokens": ["United", "Nations", "Environment", "Programme", "launches", "climate", "report", "in", "New", "York", "on", "April", "5", ",", "2025"],
        "labels": [2, 0, 0, 0, 1, 1, 1, 1, 2, 0, 1, 2, 0, 1, 0],
        "labels_tags": ["B", "I", "I", "I", "O", "O", "O", "O", "B", "I", "O", "B", "I", "O", "I"]
    },
    {
        "tokens": ["European", "Commission", "presents", "Green", "Deal", "roadmap", "in", "Brussels", "on", "Tuesday"],
        "labels": [2, 0, 1, 2, 0, 1, 1, 2, 1, 2],
        "labels_tags": ["B", "I", "O", "B", "I", "O", "O", "B", "O", "B"]
    },
    {
        "tokens": ["World", "Health", "Organization", "issues", "alert", "about", "new", "virus", "strain", "in", "South", "America"],
        "labels": [2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 0],
        "labels_tags": ["B", "I", "I", "O", "O", "O", "O", "O", "O", "O", "B", "I"]
    },
    {
        "tokens": ["Tesla", "announces", "expansion", "of", "Gigafactory", "Berlin", "on", "March", "15", ",", "2026"],
        "labels": [2, 1, 1, 1, 2, 0, 1, 2, 0, 1, 2],
        "labels_tags": ["B", "O", "O", "O", "B", "I", "O", "B", "I", "O", "B"]
    }
]

# iv) SNLI: Sentence Relations and Truth Values
# label meanings: 1: entails, 2: contradicts 0: neutral
SNLI_dataset = [
    {
        "sentences": [
            "A man inspects the uniform of a figure in some East Asian country.",
            "The man is sleeping.",
            "A soccer game with multiple males playing.",
            "Some men are playing a sport."
        ],
        "relation_labels": [2, 0, 0, 0, 0, 1],
        "sentence_labels": [1, 0, 1, 1]
    },
    {
        "sentences": [
            "A woman is reading a book on a park bench.",
            "A person is sitting outside and looking at a book.",
            "The woman is running a marathon.",
            "The park is empty."
        ],
        "relation_labels": [1, 0, 2, 0, 2, 0],
        "sentence_labels": [1, 1, 0, 0]
    },
    {
        "sentences": [
            "A man is riding a bicycle down a hill.",
            "A man is riding a vehicle.",
            "A man is pushing a bicycle up a hill.",
            "There is no bicycle anywhere nearby."
        ],
        "relation_labels": [1, 2, 2, 0, 0, 2],
        "sentence_labels": [1, 1, 0, 0]
    },
    {
        "sentences": [
            "A dog is sleeping on the couch.",
            "An animal is resting indoors.",
            "The dog is running around in the yard.",
            "There are no animals inside the house."
        ],
        "relation_labels": [1, 2, 2, 0, 2, 0],
        "sentence_labels": [1, 1, 0, 0]
    },
    {
        "sentences": [
            "Several people are waiting at a bus stop.",
            "A crowd is standing by the road to catch a bus.",
            "The street is completely empty with no one around.",
            "Some people are walking through a shopping mall."
        ],
        "relation_labels": [1, 2, 0, 2, 0, 0],
        "sentence_labels": [1, 1, 0, 1]
    }
]

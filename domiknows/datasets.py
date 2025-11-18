

amazon_reviews_dataset = [
{
  "label": [0],
  "title": ["Completely useless"],
  "text": ["The cable didn’t work at all. My phone never recognized it, and it felt flimsy right out of the box. Total waste of money."]
},
{
  "label": [1],
  "title": ["Below expectations"],
  "text": ["The blender works, but it struggles with anything harder than a banana. Very loud and the plastic feels cheap. I expected better for the price."]
},
{
  "label": [2],
  "title": ["Just okay"],
  "text": ["The keyboard is fine for basic typing, but the keys are a bit stiff and the backlight is uneven. Not terrible, but nothing special either."]
},
{
  "label": [3],
  "title": ["Pretty good overall"],
  "text": ["The vacuum has strong suction and is easy to maneuver. The dust container is a bit small, but for the price it’s a solid choice."]
},
{
  "label": [4],
  "title": ["Exceeded my expectations"],
  "text": ["Amazing sound quality for such a small speaker. Battery life is great, Bluetooth connects instantly, and it feels very well made. I’d buy it again."]
},
]

SNLI_dataset = [
    {
        "sentences": [
            "A man inspects the uniform of a figure in some East Asian country.	",
            "The man is sleeping",
            "A soccer game with multiple males playing.",
            "Some men are playing a sport."
        ],
        "relation_labels": [1,0,0,0,0,2],
        "sentence_labels": [1,0,1,1]
    }
]

"""
ECE
 ├── Electricity
 ├── Digital control
 └── Operational amplifier

Psychology
 ├── Attention
 ├── Child abuse
 ├── Social cognition
 └── Depression
 
 biochemistry
 ├── Polyermase chain reaction
 ├── Molecular biology
 ├── Northern blotting
 └── Immunology
"""

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
]

#IOB
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
    }
]
"""
disease_info_india.py - Knowledge base for the Indian plant species.

Merged into disease_info.py at import time. Each plant has a local
(Marathi) name, an English name and a scientific name.

IMPORTANT: an entry here does NOT make the AI recognise a plant. Only plants
whose classes are inside the trained model (models/model_info.json) can be
detected. The app and the About page check the model's class list, so the
entries for plants without training data (marked "trained": False) are only
used once a dataset for them is added and the model is retrained.
"""

INDIAN_PLANTS = {
    "tulsi": {
        "name": "Holy Basil", "local": "Tulsi", "scientific": "Ocimum tenuiflorum", "trained": True,
        "care": [
            "Grow in full sun (at least 4-6 hours a day) in well-drained soil.",
            "Water when the top 2-3 cm of soil is dry; never leave the pot standing in water.",
            "Pinch off flower spikes (manjari) to keep the plant bushy and leafy.",
            "Feed lightly with compost or vermicompost once a month in the growing season.",
        ],
    },
    "mango": {
        "name": "Mango", "local": "Amba", "scientific": "Mangifera indica", "trained": True,
        "care": [
            "Water young trees weekly; reduce watering for mature trees from 2-3 months before flowering.",
            "Prune after harvest to remove dead, crossing and crowded branches.",
            "Apply well-rotted farmyard manure and a balanced fertiliser after the monsoon.",
            "Keep the area under the canopy clean of fallen leaves and fruit.",
        ],
    },
    "rice": {
        "name": "Rice", "local": "Dhan", "scientific": "Oryza sativa", "trained": True,
        "care": [
            "Keep a shallow, steady water level (2-5 cm) during the vegetative stage.",
            "Apply nitrogen in split doses; avoid too much nitrogen, which invites blast.",
            "Use balanced potash and zinc; deficient soils encourage brown spot.",
            "Keep bunds and channels free of grassy weeds that host pests.",
        ],
    },
    "wheat": {
        "name": "Wheat", "local": "Gahu", "scientific": "Triticum aestivum", "trained": True,
        "care": [
            "Sow on time (early to mid November in most of India) to escape late-season rust.",
            "Give the crown-root irrigation about 20-25 days after sowing; it is the most critical one.",
            "Apply nitrogen in two or three splits rather than all at sowing.",
            "Remove volunteer wheat plants and grassy weeds near the field.",
        ],
    },
    # --- Plants without training data yet (not recognised by the AI) -------
    "jamun": {"name": "Jamun", "local": "Jambh", "scientific": "Syzygium cumini", "trained": False, "care": []},
    "custard_apple": {"name": "Custard Apple", "local": "Sitafal", "scientific": "Annona squamosa", "trained": False, "care": []},
    "sapodilla": {"name": "Sapodilla", "local": "Chiku", "scientific": "Manilkara zapota", "trained": False, "care": []},
    "curry_leaf": {"name": "Curry Leaf", "local": "Kadi Pata", "scientific": "Murraya koenigii", "trained": False, "care": []},
    "rose": {"name": "Rose", "local": "Gulab Ful", "scientific": "Rosa spp.", "trained": False, "care": []},
    "hibiscus": {"name": "Hibiscus", "local": "Jaswand", "scientific": "Hibiscus rosa-sinensis", "trained": False, "care": []},
}

INDIAN_DISEASES = {
    # ------------------------------------------------------------------ Mango
    "Mango___Anthracnose": {
        "plant": "mango", "disease": "Anthracnose", "pathogen": "Fungus (Colletotrichum gloeosporioides)",
        "description": "Anthracnose is the most common mango disease in India. It attacks young leaves, flowers and fruit, especially in humid and rainy weather.",
        "symptoms": ["Irregular dark brown to black spots on leaves", "Spots merge and the dead tissue dries and tears", "Blackened flower panicles and black spots on fruit"],
        "signs": ["Dead patches often along leaf edges and tips", "Holes or 'shot-hole' look where dry tissue falls out"],
        "causes": ["Fungal spores spread by rain splash and wind", "High humidity and rain during new leaf flush and flowering", "Infected twigs and fallen leaves left under the tree"],
        "treatment": ["Prune and burn infected twigs and leaves.", "Spray a copper fungicide (for example copper oxychloride 0.3%) or carbendazim 0.1% as per label, repeating at 15-day intervals in wet weather.", "Protect flowers with sprays at panicle emergence if the disease was seen last season."],
        "prevention": ["Prune for an open canopy that dries quickly.", "Clear fallen leaves and fruit from under the tree.", "Avoid overhead irrigation."],
    },
    "Mango___Bacterial_Canker": {
        "plant": "mango", "disease": "Bacterial Canker (Bacterial Black Spot)", "pathogen": "Bacterium (Xanthomonas campestris pv. mangiferaeindicae)",
        "description": "Bacterial canker causes raised black spots on leaves, twigs and fruit and can make fruit crack and drop.",
        "symptoms": ["Small water-soaked spots that turn black and raised", "Spots with a yellow halo, often angular", "Cracked black lesions on fruit and twigs"],
        "signs": ["Spots clustered along leaf veins or edges", "Gum oozing from cracked lesions in wet weather"],
        "causes": ["Bacteria spread by rain, wind and contaminated tools", "Wounds caused by insects, wind or pruning", "Infected nursery plants"],
        "treatment": ["Remove and burn infected leaves and twigs.", "Spray copper oxychloride (0.3%) as per label; a streptocycline spray, where approved locally, can be combined.", "Disinfect pruning tools after every cut."],
        "prevention": ["Plant disease-free grafts from a reliable nursery.", "Use windbreaks to reduce wound injury.", "Avoid working among trees when leaves are wet."],
    },
    "Mango___Cutting_Weevil": {
        "plant": "mango", "disease": "Mango Leaf-Cutting Weevil", "pathogen": "Pest (Deporaus marginatus)",
        "description": "The leaf-cutting weevil is a small beetle that lays eggs in tender mango leaves and then cuts them off near the base. New flushes can be badly damaged.",
        "symptoms": ["Tender leaves cut cleanly across near the stalk", "Many cut leaves lying under the tree", "Stunted new growth on young trees"],
        "signs": ["Straight, scissor-like cuts across the leaf blade", "Small reddish-brown weevils on new flush"],
        "causes": ["Weevil activity during new leaf flush, mostly in the rainy season", "Young trees and nurseries are more affected"],
        "treatment": ["Collect and destroy the fallen cut leaves, which carry eggs and grubs.", "If damage is heavy, spray a labelled insecticide on the new flush following local agriculture department advice."],
        "prevention": ["Inspect new flushes regularly.", "Rake and destroy fallen leaves under the canopy.", "Plough or dig the soil around the tree to expose pupae."],
    },
    "Mango___Die_Back": {
        "plant": "mango", "disease": "Die Back", "pathogen": "Fungus (Lasiodiplodia theobromae)",
        "description": "Die back makes twigs and branches dry from the tip backwards. Leaves on affected shoots turn brown and fall.",
        "symptoms": ["Leaves turn brown from the tip and edges, then curl and drop", "Twigs dry from the tip downwards", "Dark streaks inside the bark of affected shoots"],
        "signs": ["Brown, scorched-looking leaves on drying shoots", "Gum oozing from branches"],
        "causes": ["Fungus entering through wounds and pruning cuts", "Stress from drought, waterlogging or nutrient shortage", "Hot weather after the rainy season"],
        "treatment": ["Cut affected twigs 5-8 cm below the dried part and burn them.", "Paint cut ends with Bordeaux paste.", "Spray copper oxychloride (0.3%) as per label after pruning."],
        "prevention": ["Keep trees well fed and watered.", "Avoid wounding the bark.", "Prune only in dry weather and seal large cuts."],
    },
    "Mango___Gall_Midge": {
        "plant": "mango", "disease": "Gall Midge", "pathogen": "Pest (Procontarinia spp. gall midges)",
        "description": "Tiny gall midge flies lay eggs in young leaves. The larvae form small raised galls that distort leaves and reduce the tree's vigour.",
        "symptoms": ["Many small raised wart-like galls on leaves", "Galls turn brown or black with age", "Distorted, curled young leaves"],
        "signs": ["Pimple-like bumps on both sides of the leaf", "Tiny exit holes in old galls"],
        "causes": ["Gall midge flies active during new flush and flowering", "Humid weather favours the pest"],
        "treatment": ["Remove and destroy heavily galled leaves.", "Spray a labelled systemic insecticide on new flush if infestation is severe, as advised locally."],
        "prevention": ["Plough or dig the soil under the tree to expose pupae.", "Monitor new flushes closely.", "Keep the orchard clean of fallen leaves."],
    },
    "Mango___Powdery_Mildew": {
        "plant": "mango", "disease": "Powdery Mildew", "pathogen": "Fungus (Oidium mangiferae)",
        "description": "Powdery mildew forms a white powder on young leaves, flowers and small fruit. It can destroy flowering and fruit set.",
        "symptoms": ["White powdery patches on leaves and flower panicles", "Leaves curl and become distorted", "Flowers and young fruit drop"],
        "signs": ["White powder that rubs off with a finger", "Brownish, dry patches where powder was earlier"],
        "causes": ["Cool nights and warm days with high humidity during flowering", "Spores spread by wind"],
        "treatment": ["Spray wettable sulphur (0.2%) at panicle emergence, then again after 10-15 days.", "In severe cases use a labelled systemic fungicide such as hexaconazole."],
        "prevention": ["Prune for good air movement.", "Remove infected panicles and leaves.", "Start protective sprays early in seasons with cool, humid nights."],
    },
    "Mango___Sooty_Mould": {
        "plant": "mango", "disease": "Sooty Mould", "pathogen": "Fungus (Capnodium spp.) growing on insect honeydew",
        "description": "Sooty mould is a black coating that grows on the sticky honeydew left by sap-sucking insects such as hoppers, scales and mealybugs. It blocks sunlight from the leaf.",
        "symptoms": ["Black, soot-like coating on the upper leaf surface", "Leaves look dirty and dull", "Weak growth because less light reaches the leaf"],
        "signs": ["Black layer that can be peeled or wiped off", "Sticky leaves and insects (hoppers, scales, mealybugs) nearby"],
        "causes": ["Honeydew from mango hoppers, scale insects or mealybugs", "Ants protecting these insects"],
        "treatment": ["Control the sap-sucking insects first (hoppers, scales, mealybugs) with a labelled insecticide or neem oil.", "Spray a starch solution (about 1 kg starch in 20 L water, boiled and cooled); it dries and peels off with the mould.", "Wash small plants with water and mild soap."],
        "prevention": ["Monitor and control hoppers during flowering.", "Band the trunk to stop ants climbing.", "Prune to keep the canopy open."],
    },
    # ------------------------------------------------------------------- Rice
    "Rice___Brown_Spot": {
        "plant": "rice", "disease": "Brown Spot", "pathogen": "Fungus (Bipolaris oryzae)",
        "description": "Brown spot is common in rice grown on poor or nutrient-deficient soils. It weakens plants and causes discoloured, light grains.",
        "symptoms": ["Small oval brown spots with grey or whitish centres", "Spots often have a yellow halo", "Many spots make leaves dry from the tip"],
        "signs": ["Spots evenly scattered over the leaf", "Dark brown spots on the grain husk"],
        "causes": ["Infected seed", "Soils low in potash, silica or nitrogen", "Drought stress and warm, humid weather"],
        "treatment": ["Spray a labelled fungicide such as mancozeb or propiconazole if spots spread at tillering or booting.", "Correct nutrient shortages with balanced fertiliser, especially potash."],
        "prevention": ["Treat seed with a fungicide before sowing.", "Use healthy seed from a reliable source.", "Keep soil fertility balanced and avoid water stress."],
    },
    "Rice___Hispa": {
        "plant": "rice", "disease": "Rice Hispa", "pathogen": "Pest (Dicladispa armigera, a spiny beetle)",
        "description": "Rice hispa is a small, spiny, dark blue-black beetle. Adults scrape the leaf surface and larvae tunnel inside the leaf, leaving white streaks.",
        "symptoms": ["White parallel streaks along the leaf where the surface was scraped", "Irregular whitish blotches from larvae mining inside the leaf", "Leaves dry and turn papery white"],
        "signs": ["Small spiny black beetles on leaves", "Mined leaves with translucent patches"],
        "causes": ["Beetle infestations, often in the rainy season", "Heavy nitrogen fertiliser and dense planting"],
        "treatment": ["Clip and destroy leaf tips carrying grubs before transplanting.", "Sweep beetles with a hand net in small plots.", "If damage is high, spray a labelled insecticide following local agriculture department advice."],
        "prevention": ["Avoid excess nitrogen.", "Remove grassy weeds that host the beetle.", "Monitor the crop weekly in the early stages."],
    },
    "Rice___Leaf_Blast": {
        "plant": "rice", "disease": "Leaf Blast", "pathogen": "Fungus (Magnaporthe oryzae)",
        "description": "Blast is one of the most destructive rice diseases. Leaf blast weakens seedlings and young plants and can spread to the neck of the panicle.",
        "symptoms": ["Spindle- or diamond-shaped spots with grey centres and brown edges", "Spots grow and merge, killing whole leaves", "Seedlings may look burnt"],
        "signs": ["Pointed, eye-shaped lesions", "Greyish growth on lesions in humid weather"],
        "causes": ["Spores carried by wind", "Cool nights, heavy dew and high humidity", "Excess nitrogen fertiliser"],
        "treatment": ["Spray tricyclazole or another labelled blast fungicide at first symptoms, repeating as directed.", "Stop extra nitrogen top-dressing while the disease is active."],
        "prevention": ["Grow blast-resistant varieties.", "Apply nitrogen in split doses.", "Treat seed before sowing and remove infected stubble."],
    },
    # ------------------------------------------------------------------ Wheat
    "Wheat___Leaf_Rust": {
        "plant": "wheat", "disease": "Leaf Rust (Brown Rust)", "pathogen": "Fungus (Puccinia triticina)",
        "description": "Leaf rust is the most widespread wheat rust in India. Orange-brown pustules on the leaves reduce grain filling.",
        "symptoms": ["Small round orange-brown pustules scattered on the upper leaf surface", "Leaves yellow and dry early", "Shrivelled grains in severe cases"],
        "signs": ["Rust-coloured powder that rubs off on fingers", "Pustules not arranged in stripes (unlike yellow rust)"],
        "causes": ["Wind-blown spores from infected fields", "Moderate temperatures (15-25 °C) with dew or light rain"],
        "treatment": ["Spray propiconazole or tebuconazole as per label when pustules first appear.", "Repeat after 15 days if the infection continues."],
        "prevention": ["Grow rust-resistant varieties recommended for your zone.", "Sow on time; late sowing increases rust risk.", "Avoid excess nitrogen and remove volunteer wheat plants."],
    },
    # ------------------------------------------------------------------ Tulsi
    "Tulsi___Unhealthy": {
        "plant": "tulsi", "disease": "Unhealthy Leaves (cause not identified)", "pathogen": None,
        "description": "The AI sees signs of an unhealthy Tulsi plant, such as yellowing, drying, wilting or spotted leaves. Its training data marks plants only as healthy or unhealthy, so it cannot name the exact disease. Check the leaves and growing conditions using the lists below.",
        "symptoms": ["Yellowing or browning leaves", "Drooping or wilting stems", "Dry, curled or spotted leaves", "Sparse foliage and woody stems"],
        "signs": ["Brown or black spots (fungal leaf spot or blight)", "Grey fuzzy growth under leaves (downy mildew)", "Sticky leaves, webbing or tiny insects (aphids, mites, whiteflies)", "Blackened stem base (root or collar rot)"],
        "causes": ["Over-watering or poor drainage, which leads to root rot", "Too little water or strong afternoon heat", "Fungal leaf spot, blight or downy mildew in humid weather", "Sap-sucking pests", "Old plants that have flowered heavily"],
        "treatment": ["Remove yellow, spotted or dried leaves and stems.", "Let the soil dry before watering again; improve drainage if water stands.", "Spray neem oil (about 5 ml per litre with a drop of soap) for pests, every 7 days.", "For leaf spots or mildew, use a copper fungicide or a labelled organic fungicide as directed.", "If the stem base is black and soft, take healthy cuttings and replant in fresh soil."],
        "prevention": ["Plant in well-drained soil and a pot with drainage holes.", "Water at the base in the morning.", "Give 4-6 hours of sun and good air flow.", "Pinch off flower spikes regularly."],
    },
}

INDIAN_HEALTHY = {
    "Tulsi___healthy": "tulsi",
    "Mango___healthy": "mango",
    "Rice___healthy": "rice",
    "Wheat___healthy": "wheat",
}

# Class trained on photos of OTHER plants, so the model can say "not supported"
UNSUPPORTED_CLASS = "Other___unsupported_plant"

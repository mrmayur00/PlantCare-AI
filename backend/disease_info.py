"""
disease_info.py - Disease information knowledge base.

Covers every class of the PlantVillage dataset (38 classes, 14 plants).
The model's raw label (for example "Tomato___Early_blight") is matched to
an entry here with a forgiving lookup, so label spellings from different
PlantVillage versions ("Corn_(maize)___Common_rust_", "Tomato_Early_blight",
"Pepper,_bell___Bacterial_spot" ...) all resolve to the same entry.

Treatment advice is general guidance. Chemical products should always be
used according to their label and local agricultural extension advice.
"""
import re

# ---------------------------------------------------------------------------
# Plants: display name, scientific name and care tips for healthy leaves
# ---------------------------------------------------------------------------
PLANTS = {
    "apple": {
        "name": "Apple", "scientific": "Malus domestica",
        "care": [
            "Water deeply once a week in dry spells rather than lightly every day.",
            "Prune in late winter to open the canopy to light and air.",
            "Rake up and remove fallen leaves in autumn to reduce overwintering disease.",
            "Mulch around the base (not touching the trunk) to hold soil moisture.",
        ],
    },
    "blueberry": {
        "name": "Blueberry", "scientific": "Vaccinium corymbosum",
        "care": [
            "Keep the soil acidic (pH 4.5-5.5); pine-bark or sawdust mulch helps.",
            "Keep the root zone evenly moist; blueberry roots are shallow.",
            "Remove old, weak canes each winter to encourage new growth.",
            "Use a fertiliser made for acid-loving plants.",
        ],
    },
    "cherry": {
        "name": "Cherry", "scientific": "Prunus avium / Prunus cerasus",
        "care": [
            "Plant in well-drained soil; cherries dislike waterlogged roots.",
            "Prune after harvest in dry weather to reduce infection through cuts.",
            "Water at the base during fruit development.",
            "Clear fallen leaves and fruit from under the tree.",
        ],
    },
    "corn": {
        "name": "Corn (Maize)", "scientific": "Zea mays",
        "care": [
            "Give about 2.5 cm of water a week, especially during tasseling and silking.",
            "Side-dress with nitrogen when plants are knee-high.",
            "Plant in blocks rather than single rows for good pollination.",
            "Rotate with non-grass crops to keep leaf diseases down.",
        ],
    },
    "grape": {
        "name": "Grape", "scientific": "Vitis vinifera",
        "care": [
            "Train vines on a trellis so leaves dry quickly after rain.",
            "Prune hard in the dormant season; most fruit grows on new canes.",
            "Water deeply but infrequently; avoid wetting the leaves.",
            "Remove mummified berries and old clusters from the vine.",
        ],
    },
    "orange": {
        "name": "Orange", "scientific": "Citrus sinensis",
        "care": [
            "Water deeply when the top few centimetres of soil are dry.",
            "Feed with a citrus fertiliser containing micronutrients (zinc, iron, manganese).",
            "Inspect new flushes of growth for psyllids and other insects.",
            "Keep grass and weeds away from the trunk.",
        ],
    },
    "peach": {
        "name": "Peach", "scientific": "Prunus persica",
        "care": [
            "Prune to an open-centre shape for sunlight and airflow.",
            "Thin fruit so they are about 15-20 cm apart.",
            "Water regularly in dry weather, at the base of the tree.",
            "Avoid excess nitrogen, which encourages soft, disease-prone growth.",
        ],
    },
    "pepper_bell": {
        "name": "Bell Pepper", "scientific": "Capsicum annuum",
        "care": [
            "Keep soil evenly moist; uneven watering causes blossom-end rot.",
            "Mulch to keep soil off the leaves and moisture in the ground.",
            "Stake plants once fruit starts to form.",
            "Feed lightly with a balanced fertiliser; too much nitrogen reduces fruiting.",
        ],
    },
    "potato": {
        "name": "Potato", "scientific": "Solanum tuberosum",
        "care": [
            "Hill soil around stems as plants grow to protect tubers from light.",
            "Water consistently, about 2.5-5 cm a week, especially when tubers form.",
            "Use certified disease-free seed potatoes.",
            "Rotate: do not plant potatoes or tomatoes in the same spot for 3 years.",
        ],
    },
    "raspberry": {
        "name": "Raspberry", "scientific": "Rubus idaeus",
        "care": [
            "Remove canes that have fruited after harvest.",
            "Keep rows narrow so air moves through the canes.",
            "Water at the base and mulch to keep roots cool.",
            "Grow in well-drained soil; raspberries suffer in wet ground.",
        ],
    },
    "soybean": {
        "name": "Soybean", "scientific": "Glycine max",
        "care": [
            "Plant in warm soil (above 15 °C) for even emergence.",
            "Water during flowering and pod fill, the most sensitive stages.",
            "Rotate with corn or other non-legumes.",
            "Scout weekly for leaf spots and insect feeding.",
        ],
    },
    "squash": {
        "name": "Squash", "scientific": "Cucurbita pepo",
        "care": [
            "Water at the base in the morning so leaves stay dry.",
            "Give plants plenty of space for airflow.",
            "Mulch to keep fruit off damp soil.",
            "Remove old, yellowing lower leaves.",
        ],
    },
    "strawberry": {
        "name": "Strawberry", "scientific": "Fragaria × ananassa",
        "care": [
            "Water early in the day at the base of the plants.",
            "Mulch with straw to keep berries clean and dry.",
            "Remove old leaves after harvest and thin crowded runners.",
            "Replant beds every 3-4 years with healthy stock.",
        ],
    },
    "tomato": {
        "name": "Tomato", "scientific": "Solanum lycopersicum",
        "care": [
            "Water deeply at the base 2-3 times a week; avoid wetting the leaves.",
            "Stake or cage plants and prune lower leaves to improve airflow.",
            "Mulch to stop soil splashing onto the leaves.",
            "Rotate tomatoes, potatoes and peppers to a new bed each year.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Diseases (one entry per diseased PlantVillage class)
# ---------------------------------------------------------------------------
DISEASES = {
    "Apple___Apple_scab": {
        "plant": "apple", "disease": "Apple Scab", "pathogen": "Fungus (Venturia inaequalis)",
        "description": "Apple scab is a common fungal disease of apple leaves and fruit, worst in cool, wet springs.",
        "symptoms": ["Olive-green to brown velvety spots on leaves", "Leaves curl, yellow and drop early", "Dark, corky, cracked spots on fruit"],
        "signs": ["Spots with feathery, irregular edges", "Velvety olive fungal growth on the spots"],
        "causes": ["Spores released from infected fallen leaves in spring", "Long periods of leaf wetness in cool weather"],
        "treatment": ["Remove and destroy badly infected leaves and fruit.", "Apply a fungicide such as captan or a sulfur product from green-tip stage, following the label.", "Repeat sprays after rain during the main infection period."],
        "prevention": ["Rake and destroy fallen leaves in autumn.", "Prune to keep the canopy open and quick-drying.", "Choose scab-resistant apple varieties when planting.", "Water the soil, not the foliage."],
    },
    "Apple___Black_rot": {
        "plant": "apple", "disease": "Black Rot", "pathogen": "Fungus (Botryosphaeria obtusa)",
        "description": "Black rot attacks apple leaves, fruit and branches. On leaves it is often called frog-eye leaf spot.",
        "symptoms": ["Small purple spots that grow into brown 'frog-eye' spots with purple rims", "Rotting fruit that turns black and shrivels", "Sunken cankers on branches"],
        "signs": ["Tiny black fruiting bodies (pycnidia) in old spots", "Mummified black fruit left on the tree"],
        "causes": ["Spores from dead wood, cankers and mummified fruit", "Warm, humid weather and wounds on the tree"],
        "treatment": ["Prune out dead branches and cankers 20-30 cm below the infection.", "Remove all mummified fruit from the tree and ground.", "Apply a labelled fungicide (for example captan) during the growing season."],
        "prevention": ["Keep trees vigorous with balanced feeding and watering.", "Disinfect pruning tools between cuts.", "Remove fire-blighted or dead wood promptly."],
    },
    "Apple___Cedar_apple_rust": {
        "plant": "apple", "disease": "Cedar Apple Rust", "pathogen": "Fungus (Gymnosporangium juniperi-virginianae)",
        "description": "A rust fungus that needs both apple and juniper (cedar) trees to complete its life cycle.",
        "symptoms": ["Bright yellow-orange spots on the upper leaf surface", "Leaves may yellow and drop early", "Spots can also appear on fruit"],
        "signs": ["Small tube-like structures on the underside of leaf spots", "Orange gelatinous galls on nearby juniper trees in spring"],
        "causes": ["Spores blown from galls on junipers within a few kilometres", "Wet spring weather"],
        "treatment": ["Apply a labelled fungicide (for example myclobutanil) from pink bud stage through early summer.", "Remove heavily infected leaves where practical."],
        "prevention": ["Remove galls from nearby junipers, or avoid planting junipers close to apples.", "Plant rust-resistant apple varieties.", "Keep good airflow through the canopy."],
    },
    "Cherry___Powdery_mildew": {
        "plant": "cherry", "disease": "Powdery Mildew", "pathogen": "Fungus (Podosphaera clandestina)",
        "description": "Powdery mildew forms a white powdery coating on cherry leaves and shoots, weakening the tree.",
        "symptoms": ["White powdery patches on leaves", "Leaves curl upward and become distorted", "Young shoots may be stunted"],
        "signs": ["Powdery white fungal growth that rubs off", "Patches often start on the underside of leaves"],
        "causes": ["Warm days with humid nights", "Dense, shaded canopies with poor airflow"],
        "treatment": ["Prune and remove infected shoots.", "Apply sulfur or potassium bicarbonate sprays as directed on the label.", "Begin treatment at first sign of white patches."],
        "prevention": ["Prune for an open canopy.", "Avoid excess nitrogen fertiliser.", "Water the soil, not the leaves, and remove root suckers."],
    },
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "corn", "disease": "Gray Leaf Spot (Cercospora Leaf Spot)", "pathogen": "Fungus (Cercospora zeae-maydis)",
        "description": "Gray leaf spot is a serious leaf disease of corn in warm, humid regions and can cut yields sharply.",
        "symptoms": ["Long, narrow rectangular tan to grey lesions between leaf veins", "Lesions merge and kill large parts of the leaf", "Lower leaves are affected first"],
        "signs": ["Lesions with straight, parallel edges", "Greyish fuzzy appearance from spores in humid weather"],
        "causes": ["Fungus surviving on corn residue left on the field", "High humidity, fog or heavy dew with warm temperatures"],
        "treatment": ["Apply a labelled foliar fungicide (strobilurin or triazole) if lesions reach upper leaves before grain fill.", "Prioritise fields with susceptible hybrids."],
        "prevention": ["Rotate with non-host crops such as soybean.", "Bury or break down infected crop residue.", "Plant resistant hybrids."],
    },
    "Corn___Common_rust": {
        "plant": "corn", "disease": "Common Rust", "pathogen": "Fungus (Puccinia sorghi)",
        "description": "Common rust produces rusty pustules on corn leaves. It is usually mild but can be damaging on sweet corn.",
        "symptoms": ["Small cinnamon-brown raised pustules on both leaf surfaces", "Leaves may yellow when infection is heavy"],
        "signs": ["Powdery rust-coloured spores that rub off on fingers", "Pustules turn dark brown-black late in the season"],
        "causes": ["Wind-blown spores from other regions", "Cool temperatures (16-23 °C) with heavy dew"],
        "treatment": ["Apply a labelled fungicide early if pustules appear before tasseling on susceptible varieties.", "Monitor regularly; mild infections often need no treatment."],
        "prevention": ["Plant rust-resistant hybrids.", "Plant early so the crop matures before rust peaks.", "Avoid overhead irrigation late in the day."],
    },
    "Corn___Northern_Leaf_Blight": {
        "plant": "corn", "disease": "Northern Leaf Blight", "pathogen": "Fungus (Exserohilum turcicum)",
        "description": "Northern leaf blight causes long cigar-shaped lesions and can reduce yields when it appears before silking.",
        "symptoms": ["Long (2.5-15 cm) grey-green to tan cigar-shaped lesions", "Lesions start on lower leaves and move upward"],
        "signs": ["Dark grey spores on lesions in moist weather", "Lesions that cross leaf veins, unlike gray leaf spot"],
        "causes": ["Fungus overwintering on corn residue", "Moderate temperatures (18-27 °C) with long periods of leaf wetness"],
        "treatment": ["Apply a labelled foliar fungicide when lesions appear on the third leaf below the ear before tasseling.", "Remove severely infected leaves in small gardens."],
        "prevention": ["Grow resistant hybrids.", "Rotate crops and manage residue.", "Ensure good field drainage."],
    },
    "Grape___Black_rot": {
        "plant": "grape", "disease": "Black Rot", "pathogen": "Fungus (Guignardia bidwellii)",
        "description": "Black rot is one of the most damaging grape diseases in warm, humid climates and can destroy whole crops.",
        "symptoms": ["Reddish-brown circular leaf spots with dark borders", "Berries turn brown, then shrivel into hard black mummies"],
        "signs": ["Tiny black dots (pycnidia) inside leaf spots", "Mummified berries hanging in clusters"],
        "causes": ["Spores from mummies and infected canes", "Warm, wet weather in spring and early summer"],
        "treatment": ["Remove all mummified berries and infected leaves.", "Apply a labelled fungicide (for example mancozeb or myclobutanil) from early shoot growth until berries begin to colour."],
        "prevention": ["Prune and train vines for good airflow.", "Clean up fallen leaves and mummies before spring.", "Keep weeds down beneath vines."],
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "grape", "disease": "Esca (Black Measles)", "pathogen": "Fungal complex (Phaeomoniella, Phaeoacremonium and others)",
        "description": "Esca is a trunk disease of grapevines caused by wood-infecting fungi. Leaf symptoms are only the visible part.",
        "symptoms": ["'Tiger stripe' pattern: yellow or red areas between veins that dry and turn brown", "Small dark spots on berries", "Sudden wilting of the whole vine in hot weather"],
        "signs": ["Dark streaks or decay inside the trunk wood when cut", "Symptoms return on the same vine year after year"],
        "causes": ["Fungi entering through large pruning wounds", "Older vines and stressed vineyards are more affected"],
        "treatment": ["There is no cure; remove and destroy badly affected vines or cut back to healthy wood.", "Protect large pruning wounds with a wound sealant or labelled product."],
        "prevention": ["Prune in dry weather and avoid large cuts.", "Disinfect pruning tools.", "Avoid water stress during hot summers."],
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "grape", "disease": "Leaf Blight (Isariopsis Leaf Spot)", "pathogen": "Fungus (Pseudocercospora vitis)",
        "description": "A fungal leaf spot that causes irregular brown patches and early leaf drop, mostly late in the season.",
        "symptoms": ["Irregular dark red-brown spots on older leaves", "Spots join into large dead patches", "Early leaf fall"],
        "signs": ["Dark fungal growth on the underside of spots", "Yellow halo around some spots"],
        "causes": ["Warm, humid weather", "Infected leaf debris left in the vineyard"],
        "treatment": ["Remove infected leaves.", "Apply a copper-based or mancozeb fungicide according to the label."],
        "prevention": ["Clear fallen leaves after harvest.", "Improve airflow by canopy management.", "Avoid overhead watering."],
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "orange", "disease": "Huanglongbing (Citrus Greening)", "pathogen": "Bacterium (Candidatus Liberibacter spp.), spread by psyllid insects",
        "description": "Citrus greening is a deadly bacterial disease of citrus spread by the Asian citrus psyllid. Infected trees decline and cannot be cured.",
        "symptoms": ["Blotchy, asymmetrical yellow mottling of leaves", "Small, lopsided, bitter fruit that stays partly green", "Twig dieback and overall tree decline"],
        "signs": ["Yellowing that does not match on both halves of the leaf", "Psyllid insects or their white waxy droppings on new shoots"],
        "causes": ["Bacteria spread by the Asian citrus psyllid", "Planting infected nursery stock"],
        "treatment": ["There is no cure. Remove and destroy infected trees to protect healthy ones.", "Control psyllids with labelled insecticides or oils.", "Report suspected cases to the local agriculture department."],
        "prevention": ["Buy certified disease-free nursery trees.", "Inspect new growth regularly for psyllids.", "Keep trees well fed and watered to reduce stress."],
    },
    "Peach___Bacterial_spot": {
        "plant": "peach", "disease": "Bacterial Spot", "pathogen": "Bacterium (Xanthomonas arboricola pv. pruni)",
        "description": "Bacterial spot causes leaf spots, early leaf drop and blemished fruit on peaches and nectarines.",
        "symptoms": ["Small water-soaked spots that turn purple-brown", "Spot centres fall out leaving a 'shot-hole' look", "Leaves yellow and drop"],
        "signs": ["Angular spots limited by leaf veins", "Pitted or cracked spots on fruit"],
        "causes": ["Bacteria spread by rain splash and wind", "Warm, wet and windy weather, especially in sandy soils"],
        "treatment": ["Apply copper sprays in the dormant season and labelled bactericides during the season.", "Remove severely affected twigs."],
        "prevention": ["Plant resistant varieties.", "Keep trees vigorous but avoid excess nitrogen.", "Use windbreaks to reduce leaf injury."],
    },
    "Pepper,_bell___Bacterial_spot": {
        "plant": "pepper_bell", "disease": "Bacterial Spot", "pathogen": "Bacterium (Xanthomonas spp.)",
        "description": "Bacterial spot is a common disease of peppers that spots leaves and fruit and causes leaf drop.",
        "symptoms": ["Small water-soaked spots that turn brown with yellow halos", "Leaves yellow and fall", "Raised, scabby spots on fruit"],
        "signs": ["Spots more visible on the leaf underside", "Spots merge in wet weather"],
        "causes": ["Infected seed or transplants", "Rain splash and overhead watering spread the bacteria"],
        "treatment": ["Remove and destroy infected leaves and plants.", "Spray a copper-based bactericide, following the label, starting at first symptoms."],
        "prevention": ["Use certified disease-free seed and transplants.", "Water at the base, not overhead.", "Rotate away from peppers and tomatoes for 2-3 years.", "Do not work among wet plants."],
    },
    "Potato___Early_blight": {
        "plant": "potato", "disease": "Early Blight", "pathogen": "Fungus (Alternaria solani)",
        "description": "Early blight is a common fungal disease of potato leaves, usually starting on older, lower leaves.",
        "symptoms": ["Brown spots with concentric 'target' rings", "Yellow tissue around the spots", "Lower leaves die first"],
        "signs": ["Bullseye rings inside each spot", "Spots bounded by leaf veins"],
        "causes": ["Fungus surviving in soil and plant debris", "Warm temperatures with alternating wet and dry periods", "Stressed or poorly fed plants"],
        "treatment": ["Remove infected lower leaves.", "Apply a labelled fungicide such as chlorothalonil or mancozeb at the first sign of spots.", "Repeat as directed on the label during humid weather."],
        "prevention": ["Rotate crops for at least 2-3 years.", "Keep plants well fed, especially with nitrogen.", "Water at the base in the morning.", "Remove volunteer potatoes and debris."],
    },
    "Potato___Late_blight": {
        "plant": "potato", "disease": "Late Blight", "pathogen": "Oomycete (Phytophthora infestans)",
        "description": "Late blight is a fast-spreading, destructive disease (the cause of the Irish potato famine) that can kill plants within days.",
        "symptoms": ["Large dark, water-soaked patches on leaves", "Patches spread quickly in cool, wet weather", "Brown rot in tubers"],
        "signs": ["White fuzzy growth on leaf undersides at patch edges in humid conditions", "Foul smell from rotting plants"],
        "causes": ["Spores carried by wind and rain from infected plants", "Cool (10-20 °C), wet, humid weather", "Infected seed tubers"],
        "treatment": ["Remove and destroy infected plants immediately; do not compost them.", "Protect healthy plants with a labelled fungicide (for example mancozeb or chlorothalonil).", "Harvest tubers only after the foliage is dead, in dry weather."],
        "prevention": ["Plant certified disease-free seed potatoes.", "Space plants for good airflow.", "Avoid overhead irrigation.", "Destroy cull piles and volunteer plants."],
    },
    "Squash___Powdery_mildew": {
        "plant": "squash", "disease": "Powdery Mildew", "pathogen": "Fungus (Podosphaera xanthii and others)",
        "description": "Powdery mildew is very common on squash and pumpkins and reduces yield and fruit quality.",
        "symptoms": ["White powdery spots on leaves and stems", "Spots spread to cover the whole leaf", "Leaves yellow, brown and die"],
        "signs": ["Talcum-like white growth that rubs off", "Starts on older leaves"],
        "causes": ["Warm days, cool nights and high humidity", "Crowded plants and shade"],
        "treatment": ["Remove the worst-affected leaves.", "Spray sulfur, potassium bicarbonate or neem oil according to the label.", "Treat early and repeat every 7-14 days."],
        "prevention": ["Grow resistant varieties.", "Space plants widely and plant in full sun.", "Water at the base in the morning."],
    },
    "Strawberry___Leaf_scorch": {
        "plant": "strawberry", "disease": "Leaf Scorch", "pathogen": "Fungus (Diplocarpon earlianum)",
        "description": "Leaf scorch causes many small purple spots that make strawberry leaves look burnt.",
        "symptoms": ["Many small purple to reddish spots", "Spots merge and leaves look scorched and dry", "Weakened plants and fewer berries"],
        "signs": ["Spots without pale centres, unlike common leaf spot", "Tiny dark fungal dots in older spots"],
        "causes": ["Fungus surviving on old infected leaves", "Wet weather and overhead watering"],
        "treatment": ["Remove and destroy infected leaves.", "Apply a labelled fungicide (for example captan) during bloom if the disease is severe."],
        "prevention": ["Plant resistant varieties.", "Renovate beds after harvest by mowing old leaves.", "Use drip irrigation and keep good spacing."],
    },
    "Tomato___Bacterial_spot": {
        "plant": "tomato", "disease": "Bacterial Spot", "pathogen": "Bacterium (Xanthomonas spp.)",
        "description": "Bacterial spot causes small dark spots on tomato leaves and fruit and spreads quickly in warm, wet weather.",
        "symptoms": ["Small dark, greasy-looking spots on leaves", "Spots may have yellow halos", "Raised scabby spots on fruit"],
        "signs": ["Spots under 3 mm across, often angular", "Leaf edges may brown and tear"],
        "causes": ["Infected seed and transplants", "Rain splash, overhead watering and handling wet plants"],
        "treatment": ["Remove infected leaves and badly affected plants.", "Apply a copper-based bactericide as directed on the label."],
        "prevention": ["Use disease-free seed and transplants.", "Water at the base and avoid working among wet plants.", "Rotate crops for 2-3 years."],
    },
    "Tomato___Early_blight": {
        "plant": "tomato", "disease": "Early Blight", "pathogen": "Fungus (Alternaria solani)",
        "description": "Early blight is a fungal disease of tomatoes that usually starts on older lower leaves and moves upward.",
        "symptoms": ["Dark brown circular spots with concentric rings", "Yellowing around the spots", "Lower leaves die and drop"],
        "signs": ["Bullseye 'target' pattern inside spots", "Dark sunken spots on stems near the soil"],
        "causes": ["Fungus surviving in soil and plant debris", "Warm, humid weather and soil splashing onto leaves"],
        "treatment": ["Remove infected leaves.", "Avoid overhead watering.", "Use an appropriate fungicide such as chlorothalonil or copper, following the label."],
        "prevention": ["Maintain proper plant spacing.", "Mulch to prevent soil splash.", "Remove infected plant debris after the season.", "Rotate crops yearly."],
    },
    "Tomato___Late_blight": {
        "plant": "tomato", "disease": "Late Blight", "pathogen": "Oomycete (Phytophthora infestans)",
        "description": "Late blight is a fast, destructive disease of tomatoes and potatoes that thrives in cool, wet weather.",
        "symptoms": ["Large greasy grey-green to brown patches on leaves", "Dark brown streaks on stems", "Firm brown rot on fruit"],
        "signs": ["White fuzzy growth on leaf undersides in humid conditions", "Rapid spread across the plant in days"],
        "causes": ["Wind-blown spores from infected tomatoes or potatoes", "Cool, wet, humid weather"],
        "treatment": ["Remove and destroy infected plants; do not compost them.", "Protect nearby plants with a labelled fungicide (chlorothalonil or copper)."],
        "prevention": ["Plant resistant varieties.", "Keep leaves dry and space plants well.", "Do not grow tomatoes near potatoes.", "Remove volunteer plants."],
    },
    "Tomato___Leaf_Mold": {
        "plant": "tomato", "disease": "Leaf Mold", "pathogen": "Fungus (Passalora fulva)",
        "description": "Leaf mold is a fungal disease mostly found in greenhouses and humid areas with poor airflow.",
        "symptoms": ["Pale green to yellow spots on upper leaf surface", "Leaves curl, wither and drop"],
        "signs": ["Olive-green to brown velvety mould on the underside of leaves"],
        "causes": ["Relative humidity above 85%", "Poor ventilation, especially in greenhouses"],
        "treatment": ["Remove infected leaves.", "Improve ventilation and lower humidity.", "Apply a labelled fungicide if needed."],
        "prevention": ["Space and prune plants for airflow.", "Water at the base in the morning.", "Use resistant varieties in greenhouses."],
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "tomato", "disease": "Septoria Leaf Spot", "pathogen": "Fungus (Septoria lycopersici)",
        "description": "Septoria leaf spot causes many small spots and heavy leaf loss, weakening tomato plants.",
        "symptoms": ["Many small round spots with dark borders and grey-white centres", "Lower leaves yellow and fall"],
        "signs": ["Tiny black dots (fruiting bodies) in spot centres"],
        "causes": ["Fungus on old tomato debris and weeds", "Rain splash and warm, wet weather"],
        "treatment": ["Remove infected lower leaves.", "Apply a labelled fungicide such as chlorothalonil or copper."],
        "prevention": ["Mulch and avoid overhead watering.", "Rotate crops and clear debris.", "Control nightshade weeds."],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "tomato", "disease": "Spider Mites (Two-spotted Spider Mite)", "pathogen": "Pest (Tetranychus urticae)",
        "description": "Two-spotted spider mites are tiny pests that suck sap from leaves, causing stippling and bronzing.",
        "symptoms": ["Fine yellow or white speckles (stippling) on leaves", "Leaves turn bronze, dry and fall", "Reduced plant vigour"],
        "signs": ["Fine webbing on leaf undersides", "Tiny moving dots visible with a magnifier"],
        "causes": ["Hot, dry, dusty conditions", "Overuse of broad-spectrum insecticides that kill natural predators"],
        "treatment": ["Spray leaf undersides with a strong jet of water.", "Use insecticidal soap, neem oil or a labelled miticide.", "Release predatory mites in greenhouses."],
        "prevention": ["Keep plants well watered to reduce stress.", "Check leaf undersides regularly in hot weather.", "Avoid unnecessary insecticide sprays."],
    },
    "Tomato___Target_Spot": {
        "plant": "tomato", "disease": "Target Spot", "pathogen": "Fungus (Corynespora cassiicola)",
        "description": "Target spot causes brown ringed spots on tomato leaves, stems and fruit in warm, humid climates.",
        "symptoms": ["Brown spots with light centres and concentric rings", "Spots merge and leaves die", "Sunken spots on fruit"],
        "signs": ["Ringed 'target' pattern", "Spots often surrounded by a yellow halo"],
        "causes": ["Warm (20-30 °C), humid weather", "Dense canopies and long leaf wetness"],
        "treatment": ["Remove infected leaves.", "Apply a labelled fungicide (for example chlorothalonil)."],
        "prevention": ["Prune lower leaves for airflow.", "Rotate crops and remove debris.", "Avoid overhead watering."],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "tomato", "disease": "Tomato Yellow Leaf Curl Virus", "pathogen": "Virus (TYLCV), spread by whiteflies",
        "description": "A viral disease spread by whiteflies that stunts tomato plants and severely reduces yield.",
        "symptoms": ["Upward curling and cupping of leaves", "Yellow leaf edges", "Stunted plants and flower drop"],
        "signs": ["Small white flying insects (whiteflies) under leaves", "New growth is small and crumpled"],
        "causes": ["Virus carried by the silverleaf whitefly", "Infected transplants"],
        "treatment": ["There is no cure; remove and destroy infected plants.", "Control whiteflies with sticky traps, insecticidal soap or labelled insecticides."],
        "prevention": ["Plant resistant varieties.", "Use insect-proof netting for seedlings.", "Remove weeds that host whiteflies."],
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "tomato", "disease": "Tomato Mosaic Virus", "pathogen": "Virus (ToMV)",
        "description": "A very stable virus spread mainly by touch, tools and infected seed. It causes mottled, distorted leaves.",
        "symptoms": ["Light and dark green mosaic pattern on leaves", "Leaves may be curled, narrow or fern-like", "Stunted growth and uneven fruit ripening"],
        "signs": ["Mottling is most visible on young leaves"],
        "causes": ["Contaminated hands, tools and stakes", "Infected seed and plant debris"],
        "treatment": ["There is no cure; remove and destroy infected plants.", "Disinfect tools and wash hands after handling plants."],
        "prevention": ["Use certified virus-free seed.", "Do not use tobacco products near plants.", "Grow resistant varieties."],
    },
}

# Healthy PlantVillage classes -> plant key
HEALTHY = {
    "Apple___healthy": "apple",
    "Blueberry___healthy": "blueberry",
    "Cherry___healthy": "cherry",
    "Corn___healthy": "corn",
    "Grape___healthy": "grape",
    "Peach___healthy": "peach",
    "Pepper,_bell___healthy": "pepper_bell",
    "Potato___healthy": "potato",
    "Raspberry___healthy": "raspberry",
    "Soybean___healthy": "soybean",
    "Strawberry___healthy": "strawberry",
    "Tomato___healthy": "tomato",
}

GENERAL_CARE = [
    "Check plants every few days so problems are caught early.",
    "Remove fallen and dead leaves to stop diseases building up.",
    "Use clean tools and wash hands after handling sick plants.",
]

# ---------------------------------------------------------------------------
# Indian plant species (Tulsi, Mango, Rice, Wheat, ...) from disease_info_india.py
# ---------------------------------------------------------------------------
from backend.disease_info_india import (  # noqa: E402
    INDIAN_DISEASES, INDIAN_HEALTHY, INDIAN_PLANTS, UNSUPPORTED_CLASS)

PLANTS.update(INDIAN_PLANTS)
DISEASES.update(INDIAN_DISEASES)
HEALTHY.update(INDIAN_HEALTHY)

SUPPORTED_LIST_TEXT = (
    "Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, "
    "Soybean, Squash, Strawberry, Tomato, Tulsi, Mango (Amba), Rice (Dhan) and Wheat (Gahu)"
)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------
def _compact(label):
    """
    Turn any label spelling into a simple comparison key.

    "Corn_(maize)___Common_rust_"   -> "corncommonrust"
    "Pepper,_bell___Bacterial_spot" -> "pepperbellbacterialspot"
    """
    label = re.sub(r"\(.*?\)", "", label)          # drop text in brackets
    return re.sub(r"[^a-z0-9]", "", label.lower())  # keep letters and digits


# Pre-built index: compact key -> ("disease", key) or ("healthy", plant)
_INDEX = {}
for _key in DISEASES:
    _INDEX[_compact(_key)] = ("disease", _key)
for _key, _plant in HEALTHY.items():
    _INDEX[_compact(_key)] = ("healthy", _plant)


def _guess_plant_name(label):
    """Best-effort plant name for a label not in this knowledge base."""
    first = re.split(r"_{2,}|,", label)[0]
    return re.sub(r"\(.*?\)", "", first).replace("_", " ").strip().title() or "Unknown plant"


def get_result_info(label):
    """
    Build the full information block for a model label.

    Returns a dict with plant, scientific_name, disease, status,
    and the information lists shown on the result page.
    """
    if _compact(label) == _compact(UNSUPPORTED_CLASS):
        # The model recognised the leaf as one of the "other plants" it was shown
        return {
            "plant": "Not a supported plant",
            "local_name": None,
            "scientific_name": None,
            "disease": "Not identified",
            "status": "Uncertain",
            "pathogen": None,
            "description": (
                "This leaf does not look like any plant species the AI was trained on, so it "
                "cannot give a reliable disease result. Supported plants: "
                + _supported_text() + "."),
            "symptoms": [], "signs": [], "causes": [], "treatment": [],
            "prevention": [
                "Check that the plant is one of the supported species listed on the About page.",
                "Photograph a single leaf filling most of the frame, in daylight.",
            ],
        }

    match = _INDEX.get(_compact(label))

    if match and match[0] == "healthy":
        plant = PLANTS[match[1]]
        return {
            "plant": plant["name"],
            "local_name": plant.get("local"),
            "scientific_name": plant["scientific"],
            "disease": "None (Healthy)",
            "status": "Healthy",
            "pathogen": None,
            "description": f"The leaf looks healthy. No signs of the diseases this model knows for {plant['name'].lower()} were found.",
            "symptoms": [], "signs": [], "causes": [], "treatment": [],
            "prevention": plant["care"] + GENERAL_CARE,
        }

    if match and match[0] == "disease":
        d = DISEASES[match[1]]
        plant = PLANTS[d["plant"]]
        return {
            "plant": plant["name"],
            "local_name": plant.get("local"),
            "scientific_name": plant["scientific"],
            "disease": d["disease"],
            "status": "Diseased",
            "pathogen": d["pathogen"],
            "description": d["description"],
            "symptoms": d["symptoms"],
            "signs": d["signs"],
            "causes": d["causes"],
            "treatment": d["treatment"],
            "prevention": d["prevention"] + plant["care"][:2],
        }

    # Label is supported by the model but missing from this knowledge base
    healthy = "healthy" in label.lower()
    disease_part = re.split(r"_{3}", label)[-1] if "___" in label else label
    return {
        "plant": _guess_plant_name(label),
        "local_name": None,
        "scientific_name": None,
        "disease": "None (Healthy)" if healthy else disease_part.replace("_", " ").strip().title(),
        "status": "Healthy" if healthy else "Diseased",
        "pathogen": None,
        "description": "Detailed information for this class has not been added to the knowledge base yet.",
        "symptoms": [], "signs": [], "causes": [], "treatment": [],
        "prevention": GENERAL_CARE,
    }


def display_plant_name(info):
    """'Mango (Amba)' style name for lists; plain name if there is no local name."""
    if info.get("local_name"):
        return f"{info['plant']} ({info['local_name']})"
    return info["plant"]


# Class names of the loaded model (set by app.py). Only these are really
# recognised by the AI, whatever else this knowledge base contains.
_model_classes = []


def set_model_classes(class_names):
    global _model_classes
    _model_classes = list(class_names)


def _supported_text():
    plants = sorted({display_plant_name(get_result_info(c)) for c in _model_classes
                     if _compact(c) != _compact(UNSUPPORTED_CLASS)})
    return ", ".join(plants) if plants else SUPPORTED_LIST_TEXT


def supported_classes(model_classes=None):
    """
    (plant, disease) pairs the AI can actually detect.

    With the model's class list, only classes inside the model are returned.
    Without it (model not loaded), every entry of the knowledge base is listed.
    """
    if model_classes:
        rows = []
        for c in model_classes:
            info = get_result_info(c)
            if info["status"] in ("Healthy", "Diseased"):
                rows.append((display_plant_name(info), "Healthy" if info["status"] == "Healthy" else info["disease"]))
        return sorted(rows)
    rows = [(display_plant_name({"plant": PLANTS[d["plant"]]["name"], "local_name": PLANTS[d["plant"]].get("local")}),
             d["disease"]) for d in DISEASES.values() if PLANTS[d["plant"]].get("trained", True)]
    rows += [(display_plant_name({"plant": PLANTS[p]["name"], "local_name": PLANTS[p].get("local")}), "Healthy")
             for p in HEALTHY.values() if PLANTS[p].get("trained", True)]
    return sorted(rows)


def plants_without_model_support(model_classes):
    """Plants in the knowledge base that the loaded model cannot recognise yet."""
    covered = {get_result_info(c)["plant"] for c in model_classes}
    return sorted(
        f"{p['local']} ({p['name']}) - {p['scientific']}"
        for p in PLANTS.values() if p.get("local") and p["name"] not in covered)

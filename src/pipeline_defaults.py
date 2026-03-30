# Common North American bird species for the Long Island / Northeast region.
# BioCLIP is queried against this list — narrower list = faster + more accurate.
# Add/remove species to tune for your local area.

COMMON_NA_BIRDS = [
    # Waterfowl
    "Canada Goose", "Cackling Goose", "Snow Goose", "Brant",
    "Mallard", "American Black Duck", "Northern Pintail", "Green-winged Teal",
    "Blue-winged Teal", "American Wigeon", "Northern Shoveler",
    "Canvasback", "Redhead", "Ring-necked Duck", "Greater Scaup", "Lesser Scaup",
    "Long-tailed Duck", "Bufflehead", "Common Goldeneye",
    "Hooded Merganser", "Common Merganser", "Red-breasted Merganser",
    "Ruddy Duck", "Mute Swan", "Tundra Swan",
    "Wood Duck",

    # Loons, Grebes
    "Common Loon", "Red-throated Loon", "Pied-billed Grebe",
    "Horned Grebe", "Red-necked Grebe",

    # Cormorants, Pelicans
    "Double-crested Cormorant", "Great Cormorant",
    "Brown Pelican", "American White Pelican",

    # Herons, Egrets, Ibis
    "Great Blue Heron", "Great Egret", "Snowy Egret", "Little Blue Heron",
    "Tricolored Heron", "Green Heron", "Black-crowned Night-Heron",
    "Yellow-crowned Night-Heron", "Glossy Ibis",

    # Raptors
    "Osprey", "Bald Eagle", "Northern Harrier", "Sharp-shinned Hawk",
    "Cooper's Hawk", "Red-tailed Hawk", "Red-shouldered Hawk",
    "Broad-winged Hawk", "Rough-legged Hawk", "American Kestrel",
    "Merlin", "Peregrine Falcon",
    "Turkey Vulture",

    # Rails, Coots
    "Clapper Rail", "Virginia Rail", "Sora", "American Coot",
    "Common Gallinule",

    # Shorebirds
    "Killdeer", "American Oystercatcher", "Black-bellied Plover",
    "Semipalmated Plover", "Piping Plover",
    "Greater Yellowlegs", "Lesser Yellowlegs", "Willet",
    "Spotted Sandpiper", "Ruddy Turnstone", "Dunlin",
    "Sanderling", "Semipalmated Sandpiper", "Least Sandpiper",
    "Short-billed Dowitcher", "American Woodcock", "Wilson's Snipe",

    # Gulls, Terns
    "Laughing Gull", "Ring-billed Gull", "Herring Gull",
    "Great Black-backed Gull", "Bonaparte's Gull", "Iceland Gull",
    "Lesser Black-backed Gull",
    "Forster's Tern", "Common Tern", "Least Tern", "Royal Tern",
    "Caspian Tern", "Black Skimmer",

    # Alcids
    "Common Murre", "Razorbill", "Atlantic Puffin", "Thick-billed Murre",

    # Doves, Pigeons
    "Rock Pigeon", "Mourning Dove", "Eurasian Collared-Dove",

    # Owls
    "Eastern Screech-Owl", "Great Horned Owl", "Barred Owl",
    "Short-eared Owl", "Long-eared Owl", "Snowy Owl", "Barn Owl",
    "Northern Saw-whet Owl",

    # Swifts, Hummingbirds
    "Chimney Swift", "Ruby-throated Hummingbird",

    # Kingfisher
    "Belted Kingfisher",

    # Woodpeckers
    "Red-bellied Woodpecker", "Downy Woodpecker", "Hairy Woodpecker",
    "Northern Flicker", "Pileated Woodpecker", "Yellow-bellied Sapsucker",
    "Red-headed Woodpecker",

    # Flycatchers
    "Eastern Kingbird", "Great Crested Flycatcher",
    "Eastern Phoebe", "Eastern Wood-Pewee",
    "Willow Flycatcher", "Least Flycatcher", "Alder Flycatcher",

    # Vireos
    "White-eyed Vireo", "Blue-headed Vireo", "Warbling Vireo", "Red-eyed Vireo",
    "Philadelphia Vireo",

    # Jays, Crows
    "Blue Jay", "American Crow", "Fish Crow", "Common Raven",

    # Swallows
    "Tree Swallow", "Northern Rough-winged Swallow", "Bank Swallow",
    "Cliff Swallow", "Barn Swallow", "Purple Martin",

    # Chickadees, Titmice
    "Black-capped Chickadee", "Carolina Chickadee", "Tufted Titmouse",

    # Nuthatches, Creepers
    "White-breasted Nuthatch", "Red-breasted Nuthatch", "Brown Creeper",

    # Wrens
    "House Wren", "Winter Wren", "Marsh Wren", "Sedge Wren", "Carolina Wren",

    # Kinglets, Gnatcatchers
    "Golden-crowned Kinglet", "Ruby-crowned Kinglet", "Blue-gray Gnatcatcher",

    # Thrushes
    "Eastern Bluebird", "Veery", "Swainson's Thrush", "Hermit Thrush",
    "American Robin", "Gray-cheeked Thrush", "Bicknell's Thrush",

    # Mimics
    "Gray Catbird", "Northern Mockingbird", "Brown Thrasher",

    # Starlings
    "European Starling",

    # Waxwings
    "Cedar Waxwing", "Bohemian Waxwing",

    # Warblers
    "Ovenbird", "Worm-eating Warbler", "Louisiana Waterthrush",
    "Northern Waterthrush", "Blue-winged Warbler", "Golden-winged Warbler",
    "Black-and-white Warbler", "Prothonotary Warbler",
    "Tennessee Warbler", "Orange-crowned Warbler", "Nashville Warbler",
    "Common Yellowthroat", "Hooded Warbler", "American Redstart",
    "Cape May Warbler", "Northern Parula", "Magnolia Warbler",
    "Bay-breasted Warbler", "Blackburnian Warbler", "Yellow Warbler",
    "Chestnut-sided Warbler", "Blackpoll Warbler",
    "Black-throated Blue Warbler", "Palm Warbler", "Pine Warbler",
    "Yellow-rumped Warbler", "Prairie Warbler",
    "Black-throated Green Warbler", "Yellow-throated Warbler",
    "Wilson's Warbler", "Canada Warbler",

    # Tanagers, Grosbeaks, Buntings
    "Scarlet Tanager", "Summer Tanager",
    "Northern Cardinal", "Rose-breasted Grosbeak", "Blue Grosbeak",
    "Indigo Bunting", "Painted Bunting", "Dickcissel",

    # Sparrows, Towhees
    "Eastern Towhee", "American Tree Sparrow", "Chipping Sparrow",
    "Field Sparrow", "Vesper Sparrow", "Savannah Sparrow",
    "Grasshopper Sparrow", "Henslow's Sparrow", "Nelson's Sparrow",
    "Seaside Sparrow", "Song Sparrow", "Lincoln's Sparrow",
    "Swamp Sparrow", "White-throated Sparrow", "White-crowned Sparrow",
    "Dark-eyed Junco", "Fox Sparrow",

    # Blackbirds, Orioles
    "Bobolink", "Eastern Meadowlark", "Red-winged Blackbird",
    "Brown-headed Cowbird", "Rusty Blackbird", "Common Grackle",
    "Baltimore Oriole", "Orchard Oriole",

    # Finches
    "House Finch", "Purple Finch", "Common Redpoll", "Pine Siskin",
    "American Goldfinch", "Evening Grosbeak", "Pine Grosbeak",

    # Old World sparrows
    "House Sparrow",
]

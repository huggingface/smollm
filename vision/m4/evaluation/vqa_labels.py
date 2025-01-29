# Very very dirty hack to get the answers for the VQA dataset
# List is from https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt
# Ideally, we should put these answers inside `dataset.features` as labels of the vqa datasets (if that's actually possible, need to double check...)
# If so, we will be able to call `dataset.features[self.label_column_name].names`, `dataset.features[self.label_column_name].str2int`, etc. directly without having to do this hack

string_list = """<unk>
0
000
1
1 4
1 foot
1 hour
1 in back
1 in front
1 in middle
1 inch
1 on left
1 on right
1 way
1 world
1 year
1.00
10
10 feet
10 inches
10 years
100
100 feet
100 year party ct
1000
101
106
10:00
10:05
10:08
10:10
10:15
10:20
10:25
10:30
10:35
10:40
10:45
10:50
10:55
11
11:00
11:05
11:10
11:15
11:20
11:25
11:30
11:35
11:45
11:50
11:55
12
12 feet
120
12:00
12:05
12:10
12:15
12:20
12:25
12:28
12:30
12:35
12:40
12:45
12:50
12:55
13
14
15
15 feet
150
16
17
18
19
193
1950
1950s
1980
1990
1:00
1:05
1:10
1:15
1:20
1:25
1:30
1:35
1:40
1:45
1:50
1:55
1st
2
2 feet
2 hours
2 men
2 people
2 years
2.00
20
20 feet
20 ft
200
2000
2007
2008
2009
2010
2011
2012
2013
2015
2016
21
22
23
24
25
26
27
28
29
2:00
2:05
2:10
2:15
2:20
2:25
2:30
2:35
2:40
2:45
2:50
2:55
2nd
3
3 feet
3 inches
30
30 mph
300
31
32
33
34
35
350
36
37
38
39
3:00
3:10
3:15
3:20
3:25
3:30
3:45
3:50
3:55
3rd
4
4 feet
4 ft
4 inches
4 way
40
400
41
42
43
44
45
46
47
48
49
4:00
4:05
4:15
4:20
4:30
4:35
4:40
4:45
4:50
4:55
4th of july
5
5 feet
5 ft
5 star
5 years
50
50 feet
500
51
52
53
54
55
56
59
5:00
5:05
5:10
5:15
5:18
5:25
5:30
5:40
5:45
5:50
5:55
6
6 feet
6 inches
60
600
61
64
65
66
68
6:00
6:05
6:20
6:25
6:30
6:35
6:40
6:45
7
7 eleven
70
700
72
75
7:00
7:05
7:10
7:25
7:35
7:45
7:55
8
8 feet
80
870
88
8:00
8:05
8:35
8:50
8:55
9
90
99
9:05
9:12
9:15
9:20
9:25
9:30
9:35
9:45
9:50
9:55
aa
above
above door
above sink
above stove
above toilet
abstract
accident
acer
across street
adidas
adult
adults
advertisement
africa
african
african american
after
afternoon
against wall
age
ahead
air
air canada
air conditioner
air force
air france
airplane
airplanes
airport
alaska
alcohol
alive
all
all of them
all way
alligator
almonds
alps
aluminum
am
amazon
ambulance
america
american
american airlines
american flag
amtrak
ana
analog
angel
angels
angry
animal
animals
ankle
anniversary
antelope
antenna
antique
apartment
apartments
apple
apple and banana
apples
apron
arabic
arch
arizona
arm
army
around neck
arriving
arrow
arrows
art
ascending
asia
asian
asics
asleep
asparagus
asphalt
at camera
at table
at&t
athletics
atv
audi
australia
avocado
awake
away
b
babies
baby
baby's breath
back
back left
background
backhand
backpack
backward
backwards
backyard
bacon
bad
badminton
bag
bagel
bagels
baggage claim
bags
baked
baker
bakery
baking
balance
balcony
bald
ball
balloon
balloons
balls
bamboo
banana
banana bread
banana peel
banana split
bananas
band
bandana
bank
bank of america
bar
barbed wire
barber shop
bark
barn
barrel
barrier
bars
base
baseball
baseball bat
baseball cap
baseball field
baseball game
baseball glove
baseball player
baseball uniform
basil
basket
basketball
baskets
bat
bathing
bathing suit
bathroom
bathtub
batman
bats
batter
batting
beach
beads
beagle
beanie
beans
bear
beard
bears
bed
bedroom
beef
beer
beets
before
behind
behind bench
behind bus
behind clouds
behind fence
behind woman
beige
beijing
bell
below
belt
bench
benches
bending
berries
best buy
bib
bible
bicycle
bicycles
bidet
big
big ben
bike
bike rack
biker
bikers
bikes
biking
bikini
billabong
bin
biplane
bird
bird feeder
birds
birthday
birthday cake
birthday party
black
black and blue
black and brown
black and gray
black and orange
black and pink
black and red
black and silver
black and white
black and yellow
black white
blackberry
blanket
blankets
bleachers
blender
blending
blinders
blinds
blonde
blood
blt
blue
blue and black
blue and gray
blue and green
blue and orange
blue and pink
blue and red
blue and white
blue and yellow
blue jay
blue team
blueberries
blueberry
blurry
bmw
bnsf
board
boarding
boardwalk
boat
boating
boats
bob
bone
boogie board
book
books
bookshelf
boot
boots
bored
boredom
boston
both
bottle
bottles
bottom
bottom left
bottom right
boundaries
bow
bow tie
bowl
bowling
bowls
bowtie
box
boxer
boxes
boxing
boy
boys
brace
bracelet
braid
branch
branches
brand
brass
braves
brazil
bread
breakfast
brewers
brick
bricks
bride
bridge
bridle
briefcase
bright
britain
british
british airways
broadway
broccoli
broccoli and carrots
broke
broken
bronze
broom
brown
brown and black
brown and white
brush
brushing
brushing hair
brushing her teeth
brushing his teeth
brushing teeth
bucket
bud light
budweiser
buffalo
building
buildings
bull
bulldog
bun
bundt
bunk
bunny
bunt
buoy
buoys
burger
burgers
burrito
burton
bus
bus driver
bus station
bus stop
buses
bush
bushes
business
busy
butt
butter
butterfly
button
button up
buttons
by window
c
cabbage
cabinet
cabinets
cactus
cadillac
cafe
cage
cake
cakes
calendar
calico
california
calm
camel
camera
cameraman
cameras
camo
camouflage
camper
camping
can
can't see
can't see it
can't tell
canada
candle
candles
candy
cane
cannot tell
canoe
canon
canopy
cantaloupe
cap
captivity
car
caramel
cardboard
cardinal
cardinals
cargo
carnation
carnations
carpet
carriage
carrot
carrot cake
carrots
cars
cart
cartoon
case
casserole
cast iron
castle
casual
cat
cat and dog
cat food
catch
catch ball
catch frisbee
catcher
catching
catching frisbee
catholic
cats
caucasian
cauliflower
caution
cd
cds
ceiling
celery
cell
cell phone
cell phones
cement
center
ceramic
cereal
cessna
chain
chain link
chains
chair
chairs
chalk
champagne
chandelier
charging
chase
checkerboard
checkered
checkers
cheddar
cheese
cheesecake
chef
cherries
cherry
chest
chevrolet
chevron
chevy
chicago
chicken
chihuahua
child
children
chili
chimney
china
china airlines
chinese
chips
chiquita
chocolate
choppy
chopsticks
christian
christmas
christmas tree
chrome
church
cigarette
cigarettes
cilantro
cinnamon
circle
circles
circus
cirrus
citizen
city
city bus
clams
classic
classroom
clay
clean
cleaner
cleaning
clear
cleats
climbing
clip
clock
clock tower
clocks
close
close up
closed
closet
cloth
clothes
clothing
cloud
clouds
cloudy
club
cluttered
clydesdale
cnn
coach
coal
coaster
coat
coats
cobblestone
coca cola
cocker spaniel
coconut
coffee
coffee cup
coffee maker
coffee pot
coffee table
coins
coke
cold
coleslaw
colgate
collage
collar
collie
color
colorado
colored
comcast
comfort
comforter
coming
commercial
commuter
compaq
competition
computer
computers
concentration
concert
concrete
condiments
conductor
cone
cones
conference
conference room
confused
congratulations
construction
container
continental
control
controller
controllers
converse
cook
cooked
cookie
cookies
cooking
cool
cooler
copper
copyright
cord
corgi
corn
corner
corona
cosmo
costume
cotton
couch
counter
country
countryside
couple
court
cover
cow
cowboy
cows
crafts
crane
cranes
crates
cream
crest
crib
crocs
croissant
cross
cross country
crossing
crosstown
crosswalk
crow
crown
crows
cruise ship
csx
cubs
cucumber
cucumbers
cuddling
cumulus
cup
cupcake
cupcakes
cups
curb
curious
curly
current
curtain
curtains
curved
cushion
cut
cute
cutting
cutting board
cutting cake
cutting hair
cycling
cylinder
d
dachshund
dad
daffodil
daffodils
dairy
dairy queen
daisies
daisy
dalmatian
dancing
dandelions
dark
dawn
day
day time
daytime
db
dc
dead
dead end
deck
decoration
decorative
deep
deer
defense
deli
delivery
dell
delta
denim
descending
desert
design
desk
desktop
dessert
desserts
detroit
diamond
diamonds
diesel
diet coke
different teams
digital
dim
dining
dining room
dinner
dinosaur
dip
direction
directions
dirt
dirt bike
dirty
dishes
dishwasher
disney
display
distance
do not enter
dock
dodge
dodgers
dog
dog and cat
dog bed
dog food
dog show
dogs
dole
doll
dome
domestic
don't know
don't walk
donkey
donut
donut shop
donuts
door
doorway
dots
double
double decker
doubles
dough
doughnut
doughnuts
down
down street
downhill
downtown
dr pepper
dragon
drain
drawer
drawing
dreadlocks
dress
dresser
drink
drinking
drinking water
drinks
drive
driver
driveway
driving
drums
dry
drying
drywall
ducati
duck
ducks
dugout
dump
dump truck
dunkin donuts
dusk
e
each other
eagle
ear
earbuds
earring
earrings
ears
east
easter
easton
easy
easyjet
eat
eaten
eating
egg
egg salad
eggs
eiffel tower
electric
electricity
electronics
elephant
elephants
elm
elmo
email
emergency
emirates
empty
enclosure
end
engine
england
english
entering
equestrian
europe
evening
evergreen
exhaust
exit
eyes
f
fabric
face
facebook
factory
fair
fake
fall
falling
family
fan
fancy
fans
fanta
far
far right
farm
farmer
farmers
farmers market
fashion
fast
fast food
father
faucet
feathers
fedex
fedora
feeder
feeding
feeding giraffe
feet
fell
female
fence
fern
ferris wheel
ferry
festival
feta
few
field
fighter
fighting
finch
finger
fire
fire extinguisher
fire hydrant
fire truck
firefighter
fireman
fireplace
fires
first
first base
fish
fisheye
fishing
fishing boat
flag
flags
flamingo
flashlight
flat
flat screen
flats
flickr
flip
flip flops
flip phone
floating
flood
floor
floral
florida
flour
flower
flowers
fluffy
fluorescent
fly
fly kite
flying
flying kite
flying kites
foam
focus
fog
foggy
foil
food
food processor
food truck
foot
football
footprints
for balance
for fun
for photo
for sale
ford
foreground
forehand
forest
fork
fork and knife
fork and spoon
forks
formal
formica
forward
fountain
fox
frame
france
free
freezer
freight
freightliner
french
french fries
fresh
fridge
fried
friend
friends
fries
frisbee
frisbees
frog
front
frosted
frosting
fruit
fruit salad
fruits
full
fun
fur
furniture
futon
g
game
game controller
gaming
garage
garbage
garden
garlic
gas
gas station
gate
gatorade
gazebo
ge
geese
genetics
german
german shepherd
germany
ghost
giants
ginger
giraffe
giraffe and zebra
giraffes
girl
girl on right
girls
give way
glass
glasses
glaze
glazed
globe
glove
gloves
gmc
go
goal
goalie
goat
goatee
goats
goggles
going
gold
golden gate
golden retriever
golf
gone
good
google
goose
gothic
graduation
graffiti
grandfather
granite
grape
grapefruit
grapes
grass
grassy
gravel
gravy
gray
gray and black
gray and red
gray and white
grazing
green
green and black
green and blue
green and brown
green and orange
green and red
green and white
green and yellow
green beans
greyhound
grill
grilled
grilled cheese
grind
grinding
grizzly
grocery
grocery store
ground
guitar
guitar hero
gun
gym
h
hair
hair dryer
haircut
half
half full
halloween
hallway
ham
ham and cheese
hamburger
hammer time
hammock
hand
handicap
handle
handlebars
hands
hanger
hanging
happiness
happy
happy birthday
harbor
hard
hardwood
harley
harley davidson
harness
harry potter
hat
hats
hauling
hawaii
hawaiian
hawk
hay
hazy
he isn't
he's not
head
headband
headphones
healthy
heart
hearts
heat
heater
heavy
heels
heineken
heinz
helicopter
hello kitty
helmet
helmets
herd
herding
herself
hexagon
hiding
high
high chair
high heels
highway
hiking
hill
hills
hilly
himself
hispanic
hit
hit ball
hitting
hitting ball
hockey
holding
holding baby
holding it
holding phone
holding umbrella
hollywood
home
home plate
homemade
honda
honey
hood
hoodie
horizontal
horizontally
horns
horse
horse racing
horseback riding
horses
hose
hospital
hot
hot dog
hot dogs
hot sauce
hotel
hotel room
house
houses
hp
hsbc
htc
huge
hugging
human
humans
hummingbird
hundreds
hungry
husky
hydrant
i
i don't know
ibm
ice
ice cream
icing
identification
illinois
in
in air
in back
in background
in basket
in bowl
in box
in cabbage town
in car
in corner
in cup
in field
in front
in grass
in hand
in her hand
in his hand
in middle
in motion
in sink
in sky
in snow
in stands
in street
in suitcase
in vase
in water
index
india
indian
indians
indoor
indoors
information
inside
intersection
iphone
ipod
ireland
iris
iron
island
it isn't
it's not
it's raining
italian
italy
ivory
ivy
j
jacket
jackets
jal
japan
japanese
jar
jeans
jeep
jelly
jesus
jet
jet ski
jetblue
jets
jockey
john
jones
joshua
jp morgan
juice
jump
jumped
jumping
jungle
junk
k
kangaroo
kawasaki
kayak
kayaking
kenmore
ketchup
ketchup and mustard
kettle
keyboard
keys
khaki
kia
kicking
kickstand
kid
kids
king
kissing
kitchen
kitchenaid
kite
kite flying
kite string
kites
kitesurfing
kiting
kitten
kiwi
klm
knee pads
kneeling
knife
knife and fork
knives
kodak
korean air
krispy kreme
l
la
lab
labrador
lace
lacoste
ladder
lady
ladybug
lake
lamb
lamp
lamps
land
landing
landscape
lanes
lanyard
lap
laptop
laptops
large
laughing
laundry
laying
laying down
lays
leaf
leaning
learning
leash
leather
leaves
leaving
left
left 1
left and right
left side
leg
lego
legos
legs
lemon
lemonade
lemons
leopard
letters
lettuce
lexus
lg
library
license plate
licking
lid
life
life jacket
life vest
lifeguard
lift
light
lighter
lighthouse
lighting
lights
lilac
lilies
lily
lime
limes
lines
linoleum
lion
liquor
listening
listening to music
little
little girl
living
living room
lizard
loading
lobster
log
logitech
logo
logs
london
long
long sleeve
long time
looking
looking at camera
looking at phone
looking out window
los angeles
lot
lotion
lots
love
low
lufthansa
luggage
lunch
lying down
m
mac
macaroni
machine
mack
magazine
magazines
magnet
magnets
mailbox
main
main st
main street
makeup
male
males
mall
man
man in middle
man made
man on left
man on right
man's
mane
mango
mantle
many
map
maple
maple leaf
marble
marina
mariners
mario
marker
market
maroon
married
marshmallows
mask
mat
mattress
mayo
mayonnaise
mcdonald's
me
meat
meatballs
medium
meeting
men
men's
menu
meow
mercedes
mercedes benz
messy
metal
meter
metro
mets
mexican
mexico
miami
michigan
mickey mouse
microphone
microsoft
microwave
middle
middle 1
military
milk
millions
minnie mouse
mint
mirror
mirrors
mississippi
mitsubishi
mitt
mixer
model
modern
mohawk
mom
monday
money
monitor
monkey
monster
moon
moped
more
morning
mosaic
moss
motel
mother
mother and child
motion
motocross
motor
motorbike
motorcycle
motorcycles
motorola
mound
mountain
mountain dew
mountainous
mountains
mouse
mouse pad
mouth
mouthwash
movement
movie
moving
mozzarella
mt airy
mud
muffin
muffins
mug
multi
multi colored
multicolored
multiple
mural
museum
mushroom
mushrooms
music
mustache
mustard
mutt
n
name
name tag
napkin
napkins
nasa
nathan's
national express
natural
nature
navy
neck
necklace
neither
neon
nest
net
never
new
new orleans
new york
news
newspaper
next to toilet
night
night time
nightstand
nighttime
nike
nikon
nintendo
nissan
no
no 1
no cat
no clock
no dog
no flag
no grass
no hat
no left turn
no light
no man
no number
no parking
no plate
no shirt
no sign
no smoking
no train
no water
nobody
nokia
noodles
noon
normal
north
north america
north face
nose
not
not at all
not here
not high
not in service
not likely
not long
not possible
not sure
not there
not very
notebook
notes
nothing
now
nowhere
numbers
nursing
nuts
ny
o
oak
oar
oars
obama
ocean
octagon
octopus
off
office
oil
old
older
olives
ollie
olympics
omelet
on
on beach
on bed
on bench
on bike
on boat
on building
on bus
on car
on chair
on couch
on counter
on desk
on dresser
on elephant
on floor
on fridge
on grass
on ground
on his face
on his head
on horse
on laptop
on left
on man
on motorcycle
on napkin
on phone
on pizza
on plane
on plate
on pole
on rack
on right
on road
on rock
on runway
on shelf
on shore
on sidewalk
on sign
on sink
on skateboard
on stove
on street
on suitcase
on table
on toilet
on top
on tower
on track
on tracks
on train
on tray
on tree
on wall
on water
on woman
onion
onion rings
onions
only
opaque
open
opponent
orange
orange and black
orange and blue
orange and white
orange and yellow
orange juice
oranges
orchid
oregon
organic
oriental
orioles
ostrich
ottoman
out
out of focus
outdoor
outdoors
outfield
outside
oval
oven
over
over easy
overalls
overcast
owl
owner
p
pacific
pacifier
packing
paddle
paddle boarding
paddling
paint
painted
painting
paisley
pajamas
palm
palm tree
palm trees
pan
pancake
pancakes
panda
pans
pants
paper
paper towels
papers
parachute
parade
parakeet
parasailing
pare
paris
park
parked
parking
parking garage
parking lot
parking meter
parking meters
parmesan
parmesan cheese
parrot
parrots
parsley
partly cloudy
party
passenger
passengers
pasta
pastries
pastry
pasture
patio
patterned
paved
pavement
paw
pc
peace
peach
peaches
peacock
peanut butter
peanuts
pear
pearl
peas
pedestal
pedestrian
pedestrian crossing
pedestrians
pee
peeing
pelican
pelicans
pen
pencil
penguin
penne
pens
people
pepper
pepperoni
peppers
pepsi
persian
person
petting
petting horse
philadelphia
phillies
phone
phones
photo
photograph
photographer
photography
photoshop
piano
pickle
pickles
pickup
picnic
picnic table
picture
pictures
pie
pier
pig
pigeon
pigeons
pigtails
pillow
pillows
pilot
pine
pineapple
ping pong
pink
pink and black
pink and blue
pink and white
pink and yellow
pipe
pipes
pirate
pirates
pitbull
pitch
pitcher
pitching
pizza
pizza box
pizza cutter
pizza hut
placemat
plaid
plain
plane
planes
plant
planter
plants
plaster
plastic
plastic wrap
plate
plates
platform
play
play tennis
player
players
playing
playing baseball
playing frisbee
playing game
playing soccer
playing tennis
playing video game
playing video games
playing wii
playstation
plow
plunger
pm
pocket
pockets
pointing
polar
polar bear
polar bears
pole
poles
police
police officer
polka dot
polka dots
polo
pomeranian
pond
pony
ponytail
poodle
pool
poop
pooping
poor
porcelain
porch
pork
posing
post
poster
posts
pot
potato
potato salad
potatoes
pots
pottery
powdered
powdered sugar
power
power lines
practice
prince
print
printer
privacy
private
produce
professional
prom
propeller
protection
protest
public
public market center
pug
pull
puma
pumpkin
puppy
purple
purple and white
purse
qantas
qatar
queen
quilt
r
rabbit
race
racing
rack
racket
rackets
racquet
radiator
radio
radish
raft
rail
railing
railroad crossing
rain
rainbow
raining
rainy
ram
ramp
ranch
raspberries
raspberry
raw
rays
reading
real
rear
recently
recessed
recliner
rectangle
rectangles
red
red and black
red and blue
red and gray
red and green
red and silver
red and white
red and yellow
red bull
red light
red sox
red velvet
red white and blue
red white blue
reds
referee
reflection
refrigerator
refrigerators
regular
reins
relaxing
relish
remodeling
remote
remote control
remotes
residential
restaurant
resting
ribbon
rice
ride
riding
riding bike
riding bikes
riding elephant
riding horse
riding horses
riding motorcycle
right
right 1
right hand
right side
ring
ring finger
ripe
river
road
roast beef
robe
robin
robot
rock
rocks
rocky
rodeo
rolex
roll
roman
roman numerals
roof
room
rooster
rope
rose
roses
rottweiler
rough
round
roundabout
rowing
rubber
rug
rugby
run
running
runway
rural
russia
russian
rust
rv
rye
s
sad
saddle
safari
safe
safety
sail
sailboat
sailboats
sailing
salad
salmon
salon
salt
salt and pepper
samsung
san diego
san francisco
sand
sandals
sandwich
sandwiches
santa
santa hat
sas
sauce
sauerkraut
sausage
savannah
savory
scale
scania
scarf
scenery
schnauzer
school
school bus
scissors
scooter
scrambled
scratching
screen
seafood
seagull
seagulls
seat
seattle
seaweed
second
security
sedan
seeds
selfie
selling
semi
sepia
serious
serve
serving
sesame
sesame seeds
setting
several
sewing
shade
shadow
shadows
shaking hands
shallow
shampoo
shape
shark
shaved
shearing
shed
sheep
sheepdog
sheet
sheets
shelf
shell
shells
shelter
shelves
shepherd
shih tzu
shingles
ship
shirt
shirt and tie
shirts
shoe
shoes
shop
shopping
shopping cart
shore
short
shorter
shorts
shoulder
show
shower
shower curtain
shower head
shrimp
shut
siamese
siblings
side
side of road
sidecar
sidewalk
sideways
sign
signs
silk
silver
silver and black
silver and red
silverware
singapore
singing
single
single engine
singles
sink
sitting
size
skate
skate park
skateboard
skateboarder
skateboarding
skateboards
skatepark
skating
skeleton
ski
ski boots
ski lift
ski pole
ski poles
ski resort
ski slope
skier
skiers
skiing
skirt
skis
skull
skull and crossbones
sky
skyscraper
skyscrapers
slacks
sled
sleep
sleeping
sleeve
sliced
slide
sliding
slippers
slope
slow
slow down
small
smaller
smartphone
smile
smiley face
smiling
smoke
smoking
smooth
smoothie
snake
sneakers
sniffing
snow
snowboard
snowboarder
snowboarding
snowboards
snowflakes
snowing
snowsuit
snowy
soap
soccer
soccer ball
soccer field
socks
soda
sofa
soft
softball
soldier
soldiers
solid
someone
sony
sony ericsson
soon
soup
south
southwest
space
space needle
space shuttle
spaghetti
spanish
sparrow
spatula
speaker
speakers
spectators
speed limit
spices
spider
spiderman
spinach
spiral
spoon
spoons
sports
spots
spotted
spray paint
spring
sprinkles
sprint
sprite
square
squares
squash
squatting
squirrel
st patrick's day
stability
stadium
stagecoach
stained glass
stainless steel
stairs
stand
standing
standing still
stands
star
star alliance
star wars
starbucks
staring
stars
state farm
station
statue
statues
steak
steam
steamed
steel
steeple
steering wheel
steps
stew
stick
sticker
stickers
sticks
still
stir fry
stomach
stone
stones
stool
stop
stop light
stop sign
stopped
stopping
storage
store
stork
storm
stove
straight
straight ahead
strap
straw
strawberries
strawberry
street
street light
street name
street sign
stretching
strike
string
stripe
striped
stripes
stroller
stucco
student
students
stuffed
stuffed animal
stuffed animals
style
styrofoam
sub
subway
sugar
suit
suitcase
suitcases
suits
summer
sun
sun hat
sunbathing
sunflower
sunflowers
sunglasses
sunlight
sunny
sunrise
sunset
supreme
surf
surfboard
surfboards
surfer
surfers
surfing
surprise
surprised
sushi
suspenders
suv
suzuki
swan
swans
sweat
sweatband
sweater
sweatshirt
sweet
sweet potato
swim
swim trunks
swimming
swimsuit
swing
swinging
swinging bat
swirls
swiss
switzerland
sydney
syrup
t
t shirt
t shirt and jeans
tabby
table
tablecloth
tables
tablet
tag
tags
tail
take off
taking off
taking photo
taking picture
taking pictures
taking selfie
talking
talking on phone
tall
taller
tam
tan
tank
tank top
tape
target
tarmac
tarp
tater tots
tattoo
tattoos
taxi
tea
teacher
teal
team
teddy
teddy bear
teddy bears
teeth
telephone
television
tell time
telling time
tennis
tennis ball
tennis court
tennis player
tennis racket
tennis rackets
tennis racquet
tennis shoes
tent
tents
terrier
texas
texting
thai
thailand
thanksgiving
theater
they aren't
thick
thin
thomas
thoroughbred
thousands
throw
throw ball
throw frisbee
throwing
throwing frisbee
thumb
thumbs up
tiara
tie
tie dye
ties
tiger
tigers
tile
tiled
tiles
tim hortons
time
tinkerbell
tire
tired
tires
tissue
tissues
to catch ball
to catch frisbee
to dry
to eat
to get to other side
to hit ball
to left
to right
to see
toast
toasted
toaster
toaster oven
toilet
toilet brush
toilet paper
toiletries
toilets
tokyo
tomato
tomatoes
tongs
tongue
tools
toothbrush
toothbrushes
toothpaste
toothpick
toothpicks
top
top hat
top left
top right
toronto
toshiba
tour
tourist
tow
tow truck
toward
towards
towel
towels
tower
towing
town
toy
toyota
toys
track
tracks
tractor
traffic
traffic light
traffic lights
trailer
train
train car
train station
train tracks
trains
transport
transportation
trash
trash can
travel
traveling
tray
tree
tree branch
trees
triangle
triangles
trick
tripod
triumph
trolley
tropical
tropicana
truck
trucks
trunk
trunks
tub
tube
tugboat
tulip
tulips
tuna
tunnel
turkey
turn
turn right
turning
turtle
tusks
tuxedo
tv
tv stand
twin
twins
tying tie
typing
uk
umbrella
umbrellas
umpire
unclear
under
under armour
under sink
under table
under tree
uniform
uniforms
union station
united
united states
unknown
unsure
up
uphill
upright
ups
upside down
urban
urinal
urinals
us
us air force
us airways
us airways express
us open
usa
used
using computer
using laptop
utensils
v
vacation
vaio
valentine's day
van
vanilla
vans
vase
vases
vegetable
vegetables
vegetarian
veggie
veggies
vehicles
venice
vent
verizon
vertical
very
very big
very deep
very fast
very high
very long
very old
very tall
vest
vests
victoria
victorian
video
video game
vines
virgin
virgin atlantic
visibility
visilab
visor
volkswagen
volleyball
volvo
w
waffle
wagon
waiting
wakeboard
walgreens
walk
walking
wall
wall st
wallet
wallpaper
war
warm
warmth
warning
washing
washington
washington dc
washington monument
watch
watch tv
watching
watching tv
water
water bottle
water ski
water skiing
water skis
watermark
watermelon
wave
waves
waving
wavy
wax
wax paper
weather vane
website
wedding
weeds
welcome
west
western
westin
westjet
wet
wetsuit
wetsuits
whale
wheat
wheel
wheelchair
wheelie
wheels
whipped cream
whirlpool
white
white and black
white and blue
white and brown
white and gray
white and green
white and orange
white and pink
white and red
white and yellow
white house
whole
wicker
wide
wii
wii controller
wii controllers
wii remote
wii remotes
wiimote
wild
wildebeest
willow
wilson
wind
windmill
window
window sill
windows
windowsill
windsor
windsurfing
windy
wine
wine bottle
wine glass
wine glasses
wine tasting
wing
wings
winnie pooh
winter
wire
wireless
wires
wisconsin
woman
woman's
women
women's
wood
wooden
woodpecker
woods
wool
words
work
working
worms
wreath
wrist
wristband
writing
x
xbox
y
yacht
yamaha
yankees
yard
yarn
years
yellow
yellow and black
yellow and blue
yellow and green
yellow and orange
yellow and red
yellow and white
yes
yield
yogurt
young
younger
zebra
zebra and giraffe
zebras
zig zag
zipper
zoo
zucchini"""

_VQA_ANSWERS = string_list.split()

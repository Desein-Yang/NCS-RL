#gamelist = ('Adventure', 'AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'Carnival', 'Centipede', 'ChopperCommand', 'CrazyClimber', 'Defender', 'DemonAttack', 'DoubleDunk', 'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar', 'Hero', 'IceHockey', 'Jamesbond', 'JourneyEscape', 'Kaboom','Kangaroo','Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix', 'Pitfall', 'Pong', 'Pooyan', 'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank', 'Seaquest', 'Skiing', 'Solaris', 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 'UpNDown', 'Venture', 'VideoPinball', 'WizardOfWor', 'YarsRevenge', 'Zaxxon')
# With model
# gamelist=(BattleZone Berzerk ChooperCommand Bowling DoubleDunk Enduro Seaquest Freeway)
# without model
gamelist1=('Asteroids' 'Atlantis' 'Boxing' 'Carnival')
gamelist2=('Centipede' 'CrazyClimber'  'DemonAttack')
gamelist3=('ElevatorAction' 'FishingDerby' 'Frostbite' 'Gopher' 'Gravitar')gamelist4=('Hero' 'IceHockey' 'Jamesbond' 'JourneyEscape')
gamelist5=('Kaboom' 'Kangaroo' 'Krull' 'KungFuMaster')
gamelist6=('MsPacman' 'NameThisGame' 'Phoenix')
gamelist7=('Pooyan' 'PrivateEye' 'Qbert' 'Riverraid' 'RoadRunner')
gamelist8=('Robotank' 'Skiing' 'Solaris' 'SpaceInvaders')
gamelist9=('StarGunner' 'Tennis' 'TimePilot' 'Tutankham' 'UpNDown')
gamelist10=('Venture' 'VideoPinball' 'WizardOfWor' 'YarsRevenge' 'Zaxxon')
for i in `seq 1 10`
do
    python analysis_random.py -g ${gamelist1[i]} & python analysis_random.py -g ${gamelist2[i]} & python analysis_random.py -g ${gamelist3[i]} & python analysis_random.py -g ${gamelist4[i]} & python analysis_random.py -g ${gamelist5[i]} & python analysis_random.py -g ${gamelist6[i]} & python analysis_random.py -g ${gamelist7[i]} & python analysis_random.py -g ${gamelist8[i]} & python analysis_random.py -g ${gamelist9[i]} & python analysis_random.py -g ${gamelist10[i]}
done

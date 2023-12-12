from functions.branin import Branin
from functions.alpine01 import Alpine01
from functions.ackley import Ackley
from functions.rosenbrock import Rosenbrock
from functions.eggholder import EggHolder
from functions.hartmann import Hartmann
from functions.rastrigin import Rastrigin
from functions.schwefel import Schwefel
from functions.levy import Levy
from functions.shekel import Shekel
from functions.michalewicz import Michalewicz
from functions.sixhumpcamel import SixHumpCamel
from functions.bukin import Bukin
from functions.holdertable import HolderTable
from functions.styblinskitang import StyblinskiTang

FUNCTIONS = {'Rastrigin': Rastrigin,
             'Hartmann': Hartmann,
             'EggHolder': EggHolder,
             'Rosenbrock': Rosenbrock,
             'Ackley': Ackley,
             'Alpine01': Alpine01,
             'Branin': Branin,
             'Schwefel': Schwefel,
             'Levy': Levy,
             'Shekel': Shekel,
             "Michalewicz": Michalewicz,
             "SixHumpCamel": SixHumpCamel,
             "Bukin": Bukin,
             "HolderTable": HolderTable,
             "StyblinskiTang": StyblinskiTang}

DIM = {
    'Rastrigin': [2, 4, 10, 20, 50, 100],
    'Hartmann': [3, 6],
    'EggHolder': [2],
    'Rosenbrock': [2, 4, 10, 20, 50, 100],
    'Ackley': [2, 4, 10, 20, 50, 100],
    'Alpine01': [2, 4, 10, 20, 50, 100],
    'Branin': [2],
    'Schwefel': [2, 4, 10, 20, 50, 100],
    'Levy': [2, 4, 10, 20, 50, 100],
    'Shekel': [4],
    "Michalewicz": [2, 5, 10],
    "SixHumpCamel": [2],
    "Bukin": [2],
    "HolderTable": [2],
    "StyblinskiTang": [2, 4, 10, 20, 50, 100],
}

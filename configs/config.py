"""               DATA NAMES CONFIGURATION                """
_optimization_names = ['LabyrinthSeal_WBAO_v08_GH0_000_HCH0_0', 'LabyrinthSeal_WBAO_v08_GH0_000_HCH4_8',
                       'LabyrinthSeal_WBAO_v08_GH0_125_HCH0_0', 'LabyrinthSeal_WBAO_v08_GH0_125_HCH4_8']
_appendix_name = '_CDDT'

_converged_name = lambda name: 'converged_[{0}].dat'.format(name)
_database_name = lambda name: 'database_[{0}].dat'.format(name)
_pareto_name = lambda name: 'pareto_[{0}].dat'.format(name)
_pearson_name = lambda name: 'pearson_[{0}].dat'.format(name)
_valid_name = lambda name: 'valid_[{0}].dat'.format(name)

"""                         SETTINGS                        """
DATABASE_FILE_NAME = _converged_name(_optimization_names[0])
DATABASE_PATH = '../data'

class Base_Attacker(object):
    """An Attack Base Class"""
    def __init__(self, models, norm, epsilons, perturbation):
        """ 
        Args:

            model:
                type: a list of pytorch models

            norm:
                type: str
                choices: L0, L1, L2, L_infty 
            
            epsilons:
                type: float or None

        """
        self.models = models
        self.norm =norm
        self.epsilons = epsilons
        self.perturbation = perturbation
       
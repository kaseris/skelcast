class Registry:
    def __init__(self):
        self._module_dict = dict()

    def register_module(self, cls=None, module_name=None):
        """
        A decorator to register a module.

        Args:
        -    cls (class, optional): The class to be registered.
        -    module_name (str, optional): The name under which the class will be registered. 
               Defaults to the class name if not provided.
        """

        def _register(cls):
            nonlocal module_name
            if module_name is None:
                module_name = cls.__name__
            if module_name in self._module_dict:
                raise KeyError(f"{module_name} is already registered in {self.__class__.__name__}")
            self._module_dict[module_name] = cls
            return cls

        if cls is not None:
            return _register(cls)
        else:
            return _register

    def get_module(self, module_name):
        """
        Retrieves a class by its registered name.

        Args:
        -    module_name (str): The name of the module to retrieve.
        """
        if module_name not in self._module_dict:
            raise KeyError(f"{module_name} is not registered in {self.__class__.__name__}")
        return self._module_dict[module_name]

    def __str__(self):
        return str(self._module_dict)

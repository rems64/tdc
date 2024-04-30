class Colors:
    GREEN = "\x1b[32m"
    RED = "\x1b[91m"
    GRAY = "\x1b[37m"
    YELLOW = "\x1b[33m"
    BOLDRED = "\x1b[31m"
    RESET = "\x1b[0m"


def col(string, color) -> str:
    return color+string+Colors.RESET


class Test:
    def __init__(self, name: str, id: str):
        self.name: str = name
        self.id: str = id
        self.status: int = -1

    def success(self, message: str):
        self.status = 0
        print(col(message, Colors.GRAY))
        print(col(f"\t[{self.id}] SUCCESS", Colors.GREEN))
    
    def fail(self, message: str):
        self.status = 1
        print(col(message, Colors.RED))
        print(col(f"\t[{self.id}] ERROR", Colors.RED))        
    
    def execute_test(self):
        pass


def run_test(test: type[Test]):
    test_instance = test()
    print(col(f"RUNNING TEST [{test_instance.id}]", Colors.GREEN))
    try:
        test_instance.execute_test()
    except Exception as e:
        test_instance.fail(str(e))
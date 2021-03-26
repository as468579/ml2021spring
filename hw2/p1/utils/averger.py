class Averager():
    
    def __init__(self):
        self.num = 0.0
        self.value = 0.0
    
    def add(self, value, num=1):
        self.value = (self.value * self.num + value * num) / (self.num + num)
        self.num += num

    def getValue(self):
        return self.value 


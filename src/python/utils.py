
class DataPair:
    def __init__(self , old, new ):
        self.new = new 
        self.old = old

    def swap(self):
        self.new , self.old = self.old , self.new 
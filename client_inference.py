from client_model import client_model

if __name__ == '__main__':
    inf=client_model('101.6.69.226:8000')
    inf.inference('fan.png')
    inf.inference('football.png')
    inf.inference('handwaving.png')

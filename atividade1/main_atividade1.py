import random

def questao1():
    x = int(input("digite um número"))
    if x % 2 == 0:
        print("x é par")
    else:
        print("é impar")

def questao2():
    # cálculo de IMC
    x = float(input("digite o peso, por exemplo 75.0\n"))
    y =  float(input("digite a altura, por exemplo 1.65\n"))
    z = x / (y**2)
    print("imc ",z)
    if z <= 18.5:
        print("abaixo do peso")
    elif z > 18.5 and z <= 24.9:
        print("peso normal")
    elif z >= 25 and z <= 29.9:
        print("sobrepeso")
    else:
        print("obesidade")

def questao3():
    x = float(input("digite 3 notas\n"))
    y = float(input())
    z = float(input())
    media = (x+y+z)/3
    print("média: ",media)
    if media >= 6:
        print("aprovado")
    else:
        print("reprovado")


def questao4():
    x = random.randint(1,10)
    print("número sorteado ",x,"\n")
    while True:
        y = int(input("digite um número\n"))
        if y == x:
            break
        elif y < x:
            print("digite um número maior\n")
        else:
            print("digite um número menor\n")



""""*************************************************"""
""""chame a função conforme a necessidade de teste"""
# questao1() # função ímpar ou par

# questao2() # função IMC

# questao3() # média

# questao4() # número aleatório
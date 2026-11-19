
#motion detected means, the system senses whether someone is present in the room.#

class SmartHomeAgent:
    def __init__(self, target_temp=21.0):
        self.target_temp = target_temp
        self.history = []  # Internal model: stores recent temperature history

    def perceive(self, current_temp, motion_detected):
        # Sensors provide these inputs
        self.history.append(current_temp)

        # Keep only last 3 readings
        if len(self.history) > 3:
            self.history.pop(0)

        return self.decide_action(current_temp, motion_detected)

    def decide_action(self, current_temp, motion_detected):

        # Actuator decisions based on percepts
        if not motion_detected:
            action = "Set AC/Heater to OFF (Energy Saving Mode)"

        elif current_temp > self.target_temp + 1.0:
            action = "Turn ON AC (Cooling)"

        elif current_temp < self.target_temp - 1.0:
            action = "Turn ON Heater (Heating)"

        else:
            action = "Temperature is Optimal. Fan Only."

        return action


print("========= SMART HOME AGENT =========\n")

# User enters desired room temperature
target = float(input("Enter Target Temperature (°C): "))

# Creat Agent
agent = SmartHomeAgent(target_temp=target)

# Number of sensor readings
n = int(input("How many sensor readings do you want to enter? "))

print()

for i in range(n):

    print(f"------ Sensor Reading {i+1} ------")

    current_temp = float(input("Enter Current Temperature (°C): "))

    motion = input("Is Motion Detected? (yes/no): ").strip().lower()

    if motion == "yes":
        motion_detected = True
    else:
        motion_detected = False

    # Agent decides action
    action = agent.perceive(current_temp, motion_detected)

    print("\n------ OUTPUT ------")
    print(f"Sensors -> Temperature : {current_temp}°C")
    print(f"Sensors -> Motion      : {motion_detected}")
    print(f"Actuator -> {action}")

    # Show internal model (history)
    print(f"Internal Model (Last 3 Temps): {agent.history}")
    print()




##recommendation##
print("*"*80)
print("                    welcome to recommendation assistant           ")
print("*"*80)

print("\nselect your interest")
print("1.movie")
print("2.book")
print("3.music")

choice=input("\nenter your choice(1/2/3): ")

if choice=="1":
    print("\nchoose movie genre: ")
    print("a.action")
    print("b.comedy")
    print("c.drama")
    
    genre=input("enter genre(a/b/c): ")
    
    if genre=="a":
        print("\nrecommended movie : RRR")
    elif genre=="b":
        print("\nrecommended movie : Hera Pheri")
    elif genre=="c":
        print("\nrecommended movie : Tare Zameen Par")
    else:
       print("\ninvalid choice")
       
elif choice=="2":
    print("\nchoose book category: ")
    print("a.mythology")
    print("b.fiction")
    print("c.motivation")
    
    category=input("\nenter category(a/b/c): ")
    
    if category=="a":
        print("recommended book : The Palace of Illusion")
    elif category=="b":
        print("recommended book : The Guide")
    elif category=="c":
        print("Wings of Fire")
    else:
        print("invalid category")
        
        
elif choice=="3":
    print("\nchoose music type")
    print("a.motivational")
    print("b.devotional")
    print("c.party")
    
    music = input("enter type (a/b/c) : ")
    
    if music=="a":
        print("recommended song: Chak De India")
    elif music=="b":
        print("recommended song : Achyutam Keshavam")
    elif music=="c":
        print("recommended song : kala Chashma")
    else:
        print("invalic choice")
        
else:
    print("invalid choice")
    
print("\nthank you for using recommendation assistant")




##chatbot##
def chatbot():

    print("Chatbot Started")
    print("Type 'bye' to exit")

    while True:

        user = input("You: ").lower()

        if user == "hello":
            print("Bot: Hello! How can I help you?")

        elif user == "how are you":
            print("Bot: I am fine.")

        elif user == "your name":
            print("Bot: I am an AI Chatbot.")

        elif user == "bye":
            print("Bot: Goodbye!")
            break

        else:
            print("Bot: Sorry, I don't understand.")

chatbot()

    
        


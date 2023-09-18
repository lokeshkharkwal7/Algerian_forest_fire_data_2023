global_data_member = None  # Declare a global variable

def function1():
    global global_data_member
    global_data_member = 42

def function2():
    global global_data_member
    print(f"Accessed global_data_member from function2: {global_data_member}")

# Call function1 to set global_data_member and then call function2
function1()
function2()

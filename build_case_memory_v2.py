import os

from case_memory_v2 import CaseMemoryV2



memory=CaseMemoryV2()


folder="case_memory"


for file in os.listdir(folder):


    if file.endswith(".json"):


        path=os.path.join(
            folder,
            file
        )


        memory.add_case(path)


        print(
            "[OK]",
            file
        )
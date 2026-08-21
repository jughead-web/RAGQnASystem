# -*- coding:utf-8 -*-

import os

from case_memory import CaseMemory



def main():


    memory=CaseMemory()


    case_dir="case_memory"


    for file in os.listdir(case_dir):


        if file.endswith(".txt"):


            path=os.path.join(
                case_dir,
                file
            )


            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                text=f.read()



            patient_id=file.replace(
                ".txt",
                ""
            )


            memory.add_case(

                text,

                patient_id

            )


            print(
                "[OK]",
                patient_id
            )



if __name__=="__main__":

    main()
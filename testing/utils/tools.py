from pathlib import Path
from llama_index.core.tools import FunctionTool


def save_response(response: str, country: str):
    """Function that saves the response to a text file.

    Args:
        response: the response from the LLM that MUST be saved
        country: the name of the country to which the response corresponds

    """

    path_to_save = Path.cwd().parent.parent.joinpath("tmp")
    # print(Path.cwd())

    with open(path_to_save.joinpath(f"profile_{country}.txt"), "w+", encoding="utf-8") as file:

        file.write(response)
        file.close()



if __name__=="__main__":

    test_response = """
    Kampfgruppe Bruhns
    * Ersatz Regiment 57: 1 company plus 3 SdKfz 250/1
    * Luftwaffe troops: 1 company
    * Task: Defend the Dreijenseweg and support ounterattacks

    Kampfgruppe Krafft
    * SS Panzer-Grenadier AuE 16: 2 companies
    * Task: Form a blocking line and prepare for counterattacks
    
    What else?
    
    """

    # --------------Testing------------------
    # save_response(test_response, "brit")
    save_profile_tool = FunctionTool.from_defaults(
        save_response,
        name="save_output",
        description="For saving the LLM response to a query"
    )
    save_profile_tool.call(response=test_response, country="brit")



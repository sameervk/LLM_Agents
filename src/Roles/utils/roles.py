from pathlib import Path


class Role:
    """
    Class object instantiating a player and corresponding attributes, e.g. military, government, etc.
    The attribute information is populated from the scenario by a LLM.
    """

    def __init__(self, player: str, attributes: list):

        # player
        self.player = player

        # attributes of the player to be extracted from the scenario
        self.attributes = dict.fromkeys(attributes)

    def save_to_file(self, directory: Path, scenario: str, file_suffix: str | None):
        """
        Once the attribute information is populated, this method can then be used to save the profile information
        to a file

        Args:
            directory: folder to save
            scenario: scenario used to extract the profile info
            file_suffix: any text to be appended to the file name

        """

        if file_suffix is not None:
            file_suffix = file_suffix.replace(" ","")
            if len(file_suffix)==0:
                file_suffix=None

        file_name = f"{self.player.replace(" ","")}_{scenario.replace(" ", "")}"
        file_name = file_name + f"_{file_suffix}.txt" if file_suffix else file_name+".txt"

        with open(directory.joinpath(file_name), "w+", encoding="utf-8") as file:

            file.write(f"# {self.player} Profile \n")
            file.write("\n")

            for attr, value in self.attributes.items():

                file.write(f"## {attr} \n")
                file.write(f"{value}") if value else file.write("\n")
                file.write("\n\n")

            file.close()


if __name__=="__main__":

    test_role = Role(player="British", attributes=["military", "government"])

    test_role.save_to_file(directory=Path.cwd().parent, scenario="test", file_suffix="   x  ")








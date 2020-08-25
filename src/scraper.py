import urllib.request

import pandas as pd
from bs4 import BeautifulSoup

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Scraper:

    def __init__(self, year: int, player: str):
        self.base_url = "https://www.basketball-reference.com"
        self.players_url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
        self.curr_year = year
        self.urls = None
        self.advanced_urls = {}
        self.curr_player = player

    @staticmethod
    def make_soup(url):
        page = urllib.request.urlopen(url)
        soup_data = BeautifulSoup(page, "html.parser")
        return soup_data

    def get_player_urls(self):
        table_soup = self.make_soup(self.players_url)
        urls = {}
        for player_row in table_soup.findAll("td", {"data-stat": "player"}):
            curr_attr = str(player_row.findAll("a"))
            player_url_part = curr_attr.replace('[<a href="', "").replace('</a>]', "")
            parts = player_url_part.split('">')
            final_url = parts[0].strip('.html"') + "/gamelog/" + str(self.curr_year) + "/"
            urls[parts[1]] = final_url

        return urls

    def get_advanced_urls(self):
        self.urls = self.get_player_urls()
        for k, v in self.urls.items():
            self.advanced_urls[k] = v.replace("gamelog", "gamelog-advanced")

    def get_player_data(self, advanced=False):
        if advanced:
            self.get_advanced_urls()
            urls = self.advanced_urls
        else:
            urls = self.get_player_urls()
        if self.curr_player not in urls.keys():
            raise ValueError("Player is not in player pool")
        url = urls[self.curr_player]
        final_url = self.base_url + url
        soup = self.make_soup(final_url)
        player_table = soup.find("div", {"class": "overthrow table_container"}).find("table", {
            "class": "row_summable sortable stats_table"})
        df_player = pd.read_html(str(player_table))[0]
        df_player = df_player[df_player["Rk"] != "Rk"]

        return df_player


if __name__ == '__main__':
    dfs = []
    columns = ["Unnamed: 5", "Unnamed: 7", "Tm", "Opp", "Date", "Age", "GS", "Rk", "G", "MP", "GmSc"]
    for i in range(2010, 2021):
        try:
            print("Working on year " + str(i))
            print("-" * 80)
            s = Scraper(i, "LeBron James")
            df_data = s.get_player_data()
            df_data_adv = s.get_player_data(advanced=True)
            df_final = pd.merge(df_data, df_data_adv, on=columns, how="left")
            dfs.append(df_final)
        except ValueError:
            print("Player did not play in year {}".format(str(i)))
            print("-" * 80)
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv("data/Lebron_James_advanced_all.csv")

    # print(*s.get_player_data().items(), sep="\n")

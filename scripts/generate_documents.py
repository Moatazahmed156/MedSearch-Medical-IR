from bs4 import BeautifulSoup
import requests
import json

diseases={"physical+activity","rheumatoid+arthritis","atrial","atrial+fibrillation","alzheimer's",
          "breast","breast+cancer","back+pain","low+back","blood","covid ","covid 19","covid-19",
          "lung+cancer","stem+cell","vitamin+d","diabetes","diabetes+mellitus","2+diabetes",
          "kidney+disease","side+effects","gene+expression","lupus+erythematosus",
          "endometrial","endometrial+cancer","heart+failure","risk+factors","atrial+fibrillation"
          "fatty","fatty+liver","gastric","gastric+cancer","gut+microbiota","gene+therapy",
          "mental+health","heart","heart+failure","heart+disease","head+neck","brain+injury",
          "cord+injury","myocardial+infarction","spinal+cord+injury","kidney+injury",
          "international+journal","juvenile","juvenile+idiopathic","american+journal",
          "tight+junction","kidney","chronic+kidney","kidney+injury","acute+kidney",
          "cell+lung","low+back+pain","quality+life" ,"mental","mental+health",
          "diabetes+mellitus","gut+microbiota","myocardial","head+neck","nervous",
          "nervous+system","lymph+node","neck+cancer","oxidative","oxidative+stress","ovarian",
          "ovarian+cancer","older","back+pain","physical","physical+activity","prostate",
          "prostate+cancer","quality","quality+life","sleep+quality" ,"quality+improvement",
          "qualitative","risk+factors","case+report","reheumatoid","reheumatoid+arthritis",
          "systematic+review","stem+cell","stem+cells","spinal","spinal+cord","type+2",
          "t+cell" ,"physical+therapy","type+2+diabetes","type+1","urinary","urinary+tract",
          "ulcerative","ulcerative+colitis","medical+university","weight","weight+loss",
          "wound","wound+healing","pregnant+women","fragile+x","x+syndrome","fragile+x+syndrome",
          "xi'an","xi'an+jiaotong","young","young+adults","young+adult","yellow","yellow+fever",
          "herpes+zoster","zika+virus","zhejiang","zhejiang+university","marginal+zone"}

counter=0

for disease in diseases :
    # try:
        for number in range(1,12) :
            page = requests.get(f"https://pubmed.ncbi.nlm.nih.gov/?term={disease}&page={number}")
            soup = BeautifulSoup(page.content, 'lxml')

            articles = soup.find_all("article", class_="full-docsum")

            for article in articles:
                try:
                    title_tag = article.find('a', class_="docsum-title").text.strip()
                    doc_id = article.find('span', class_='docsum-pmid').text.strip()

                    writer_all = article.find(
                        'span', class_='docsum-authors full-authors'
                    ).text.strip()

                    if ',' in writer_all:
                        writer = writer_all.split(',')[0]
                    else:
                        writer = writer_all

                    year_text = article.find(
                        'span', class_='docsum-journal-citation short-journal-citation'
                    ).text.strip()
                    year = year_text[-5:-1]

                    summary = article.find(
                        'div', class_='full-view-snippet'
                    ).text.strip()

                    link = article.find('a', class_='docsum-title').get('href')
                    full_link = "https://pubmed.ncbi.nlm.nih.gov" + link

                    # ---- second page ----
                    second_page = requests.get(full_link)
                    second_soup = BeautifulSoup(second_page.content, 'lxml')

                    second_article = second_soup.find(
                        'div', class_='abstract-content selected'
                    ).text.strip()

                    data = {
                        "title": title_tag,
                        "doc_id": doc_id,
                        "writer": writer,
                        "year": year,
                        "summary": summary,
                        "abstract": second_article,
                        "link": full_link
                    }

                    with open(
                        f"F:\\marwan\\IR 2\\Documents\\data{counter}.json",
                        "w",
                        encoding="utf-8"
                    ) as file:
                        json.dump(data, file, ensure_ascii=False, indent=4)

                    print(f"✔ Article {counter} saved successfully")
                    counter += 1

                except Exception as e:
                    print(f"❌ Error in article {counter}: {e}")
                    continue
        print(f"✔ disease {disease} saved successfully")

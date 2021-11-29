from IPython.display import display_html 

def display_results(results, section):

    space = "\xa0" * 10

    df1 = results[section]['knn'].style.set_table_attributes("style='display:inline'").set_caption('knn_{}'.format(section))
    df2 = results[section]['ab'].style.set_table_attributes("style='display:inline'").set_caption('ab_{}'.format(section))
    df3 = results[section]['rf'].style.set_table_attributes("style='display:inline'").set_caption('rf_{}'.format(section))
    df4 = results[section]['gb'].style.set_table_attributes("style='display:inline'").set_caption('gb_{}'.format(section))
    df5 = results[section]['mlp'].style.set_table_attributes("style='display:inline'").set_caption('mlp_{}'.format(section))
    df6 = results[section]['cb'].style.set_table_attributes("style='display:inline'").set_caption('cb_{}'.format(section))

    display_html(df1._repr_html_()+ space + df2._repr_html_(), raw=True)
    display_html(df3._repr_html_()+ space + df4._repr_html_(),  raw=True)
    display_html(df5._repr_html_()+ space + df6._repr_html_(),  raw=True)

    
def display_results4(results, section):

    space = "\xa0" * 10

    df1 = results[section]['knn'].style.set_table_attributes("style='display:inline'").set_caption('knn_{}'.format(section))
    df2 = results[section]['ab'].style.set_table_attributes("style='display:inline'").set_caption('ab_{}'.format(section))
    df3 = results[section]['rf'].style.set_table_attributes("style='display:inline'").set_caption('rf_{}'.format(section))
    df4 = results[section]['cb'].style.set_table_attributes("style='display:inline'").set_caption('cb_{}'.format(section))

    display_html(df1._repr_html_()+ space + df2._repr_html_(), raw=True)
    display_html(df3._repr_html_()+ space + df4._repr_html_(),  raw=True)

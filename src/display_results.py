from IPython.display import display_html 

def display_results(results, section):

    space = "\xa0" * 10

    df1 = results['knn_{}'.format(section)].style.set_table_attributes("style='display:inline'").set_caption('knn_{}'.format(section))
    df2 = results['ab_{}'.format(section)].style.set_table_attributes("style='display:inline'").set_caption('ab_{}'.format(section))

    df3 = results['rf_{}'.format(section)].style.set_table_attributes("style='display:inline'").set_caption('rf_{}'.format(section))
    df4 = results['gb_{}'.format(section)].style.set_table_attributes("style='display:inline'").set_caption('gb_{}'.format(section))

    display_html(df1._repr_html_()+ space + df2._repr_html_(), raw=True)
    display_html(df3._repr_html_()+ space + df4._repr_html_(),  raw=True)
import sys
import re


def main(text):

    # Remove the multitude of useless spans.
    text = replace_regex(r'<\/?span[^>]*>', '', text)

    # Remove &nbsp;.
    text = replace_regex(r'&nbsp;', ' ', text)

    # Put headings on one line.
    text = replace_regex(r'<h([1-5])>\n([^<]*)<\/h\1>',
                         r'\n<h\1>\2</h\1>', text)
    # Remove class divs.
    text = replace_regex(r'<\/?div[^>]*>', '', text)

    # Remove div encapsulated brs.
    text = replace_regex(r'<div>\s*<br \/>\s*<\/div>', '</ br>', text)

    # Remove brs within heading tags.
    text = replace_regex(r'<h([1-5])>\s*(<br \/>)([^<]*)<\/h\1>',
                         r'<h\1>\3</h\1>', text)

    # Remove empty tags eg <pre></pre>.
    text = replace_regex(r'<([^>]*)>\s*<\/\1>', '', text)

    # Remove unclosed HTML tags.
    text = remove_unclosed(text)

    return text


def fix_formatting(text):
    """Fixes aesthetic formatting of text"""

    # Make sure <br \> tags have one line above and below.
    text = replace_regex(r'(\s*<br \/>\s*)', r'\n\1\n', text)

    # Remove multiple \n.
    text = replace_regex(r'\n{2,}', r'\n', text)

    return text


def replace_regex(regex, repl, text):
    """Replaces all occurances of a regex pattern regex, in text"""
    replacer = re.compile(regex, re.DOTALL)
    replaced = replacer.sub(repl, text)

    return replaced


def remove_unclosed(text):
    """removed all unclosed HTML tags"""
    unclosed = find_next_unclosed(text)

    if unclosed:
        start, end = unclosed
        text = ''.join([text[:start], text[end:]])

        return remove_unclosed(text)

    return text


def find_next_unclosed(text):
    """Finds the next unclosed HTML tag"""
    tag_stack = []

    # Get an iterator of all tags in file.
    tag_regex = re.compile(r'<[^>]*>', re.DOTALL)
    tags = tag_regex.finditer(text)

    for tag in tags:
        # If it is a closing tag check if it matches the last opening tag.
        if re.match(r'<\/[^>]*>', tag.group()):
            top_tag = tag_stack[-1]

            if tag_match(top_tag.group(), tag.group()):
                tag_stack.pop()
            else:
                unclosed = tag_stack.pop()
                return (unclosed.start(), unclosed.end())
        else:
            tag_stack.append(tag)


def tag_match(tag1, tag2):
    """Checks to see if tags are open/closing tag pairs"""

    # [1:] is for opening tag (<div> -> div>),
    # [2:] for closing (</div> -> div>).
    if get_pure_tag(tag1)[1:] == get_pure_tag(tag2)[2:]:
        return True
    else:
        return False


def get_pure_tag(tag):
    """Returns the base tag with no ids, classes, etc."""
    pure_tag = re.sub(r'<(\/?\S*)[^>]*>', r'<\1>', tag)

    return pure_tag
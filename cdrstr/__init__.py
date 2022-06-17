def to_camel_case(text):
    """
    see https://stackoverflow.com/a/60978847
    :param text: text to be converted
    :return: camel case converted text
    """
    s = text.replace("-", " ").replace("_", " ")
    s = s.split()
    if len(text) == 0:
        return text
    return s[0].lower() + ''.join(i.capitalize() for i in s[1:])

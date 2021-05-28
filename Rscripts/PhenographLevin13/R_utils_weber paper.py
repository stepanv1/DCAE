from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

string = """
square <- function(x) {
    return(x^2)
}

cube <- function(x) {
    return(x^3)
}
"""

powerpack = SignatureTranslatedAnonymousPackage(string, "powerpack")

powerpack.square(3)
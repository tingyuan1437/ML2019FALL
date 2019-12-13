class MyPorterStemmer(object):
    def __init__(self):
        self.b = ""
        self.k = 0
        self.j = 0

    def stem(self, w):
        w = w.lower()
        k = len(w) - 1
        if k <= 1:
            return w

        self.b = w
        self.k = k

        self._process1ab()
        self._process1c()
        self._process2()
        self._process3()
        self._process4()
        self._process5()
        return self.b[:self.k + 1]

    def _cons(self, i):
        ch = self.b[i]
        if ch in "aeiou":
            return False
        if ch == 'y':
            return i == 0 or not self._cons(i - 1)
        return True

    def _m(self):
        i = 0
        while True:
            if i > self.j:
                return 0
            if not self._cons(i):
                break
            i += 1
        i += 1
        n = 0
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while 1:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self):
        return not all(self._cons(i) for i in range(self.j + 1))

    def _doublec(self, j):
        return j > 0 and self.b[j] == self.b[j - 1] and self._cons(j)

    def _cvc(self, i):
        if i < 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        return self.b[i] not in "wxy"

    def _ends(self, s):
        if s[-1] != self.b[self.k]:
            return False
        length = len(s)
        if length > (self.k + 1):
            return False
        if self.b[self.k - length + 1:self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s):
        self.b = self.b[:self.j + 1] + s
        self.k = len(self.b) - 1

    def _r(self, s):
        if self._m() > 0:
            self._setto(s)

    def _process1ab(self):
        if self.b[self.k] == 's':
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != 's':
                self.k -= 1
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowelinstem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._doublec(self.k):
                if self.b[self.k - 1] not in "lsz":
                    self.k -= 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _process1c(self):
        if self._ends("y") and self._vowelinstem():
            self.b = self.b[:self.k] + 'i'

    def _process2(self):
        ch = self.b[self.k - 1]
        if ch == 'a':
            if self._ends("ational"):
                self._r("ate")
            elif self._ends("tional"):
                self._r("tion")
        elif ch == 'c':
            if self._ends("enci"):
                self._r("ence")
            elif self._ends("anci"):
                self._r("ance")
        elif ch == 'e':
            if self._ends("izer"):
                self._r("ize")
        elif ch == 'l':
            if self._ends("bli"):
                self._r("ble")
            elif self._ends("alli"):
                self._r("al")
            elif self._ends("entli"):
                self._r("ent")
            elif self._ends("eli"):
                self._r("e")
            elif self._ends("ousli"):
                self._r("ous")
        elif ch == 'o':
            if self._ends("ization"):
                self._r("ize")
            elif self._ends("ation"):
                self._r("ate")
            elif self._ends("ator"):
                self._r("ate")
        elif ch == 's':
            if self._ends("alism"):
                self._r("al")
            elif self._ends("iveness"):
                self._r("ive")
            elif self._ends("fulness"):
                self._r("ful")
            elif self._ends("ousness"):
                self._r("ous")
        elif ch == 't':
            if self._ends("aliti"):
                self._r("al")
            elif self._ends("iviti"):
                self._r("ive")
            elif self._ends("biliti"):
                self._r("ble")
        elif ch == 'g':
            if self._ends("logi"):
                self._r("log")

    def _process3(self):
        ch = self.b[self.k]
        if ch == 'e':
            if self._ends("icate"):
                self._r("ic")
            elif self._ends("ative"):
                self._r("")
            elif self._ends("alize"):
                self._r("al")
        elif ch == 'i':
            if self._ends("iciti"):
                self._r("ic")
        elif ch == 'l':
            if self._ends("ical"):
                self._r("ic")
            elif self._ends("ful"):
                self._r("")
        elif ch == 's':
            if self._ends("ness"):
                self._r("")

    def _process4(self):
        ch = self.b[self.k - 1]
        if ch == 'a':
            if not self._ends("al"):
                return
        elif ch == 'c':
            if not self._ends("ance") and not self._ends("ence"):
                return
        elif ch == 'e':
            if not self._ends("er"):
                return
        elif ch == 'i':
            if not self._ends("ic"):
                return
        elif ch == 'l':
            if not self._ends("able") and not self._ends("ible"):
                return
        elif ch == 'n':
            if self._ends("ant"):
                pass
            elif self._ends("ement"):
                pass
            elif self._ends("ment"):
                pass
            elif self._ends("ent"):
                pass
            else:
                return
        elif ch == 'o':
            if self._ends("ion") and self.b[self.j] in "st":
                pass
            elif self._ends("ou"):
                pass
            else:
                return
        elif ch == 's':
            if not self._ends("ism"):
                return
        elif ch == 't':
            if not self._ends("ate") and not self._ends("iti"):
                return
        elif ch == 'u':
            if not self._ends("ous"):
                return
        elif ch == 'v':
            if not self._ends("ive"):
                return
        elif ch == 'z':
            if not self._ends("ize"):
                return
        else:
            return
        if self._m() > 1:
            self.k = self.j

    def _process5(self):
        k = self.j = self.k
        if self.b[k] == 'e':
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(k - 1)):
                self.k -= 1
        if self.b[self.k] == 'l' and self._doublec(self.k) and self._m() > 1:
            self.k -= 1

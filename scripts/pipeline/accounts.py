"""The accounts of the web page (flow_page.py --accounts): who may log in.

Nobody makes an account on the page. The accounts are handed out from here, each with a password that is made up
here and shown once. They are kept in accounts.json in the project folder (not in Git): per name a salt and the
password's hash (PBKDF2-SHA256), never the password itself.

    python scripts/pipeline/accounts.py add anna        # makes the account, prints its password once
    python scripts/pipeline/accounts.py add anna        # again: a new password for anna (the old one is gone)
    python scripts/pipeline/accounts.py list
    python scripts/pipeline/accounts.py remove anna

A change is seen by a running page at once (the file is read at every login); removing an account also ends its
logins. Who is logged in is only kept in the running page: after a restart everybody logs in again.
"""
import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time

FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "accounts.json")
NAME = r"[a-z0-9_.-]{2,30}"  # an account's name; later also its folder
ROUNDS = 300_000  # of PBKDF2: a guess costs about a fifth of a second
KEPT = 7 * 24 * 3600  # seconds a login lasts
TRIES, LOCKED = 8, 600  # so many wrong passwords in a row, then the name waits so many seconds
SESSIONS = {}  # {token: (name, until)}: who is logged in at the running page
WRONG = {}  # {name: (wrong passwords in a row, time of the last one)}
LOCK = threading.Lock()


def read():
    try:
        with open(FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def write(accounts):
    with open(FILE + ".new", "w", encoding="utf-8") as f:
        json.dump(accounts, f, indent=1)
    os.chmod(FILE + ".new", 0o600)  # other users of the server can't read it
    os.replace(FILE + ".new", FILE)


def hashed(password, salt, rounds=None):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), rounds or ROUNDS).hex()


def add(name, password=None):
    """Makes the account (or gives an old one a new password). Returns the password: it is not kept anywhere."""
    name = name.strip().lower()
    if not re.fullmatch(NAME, name):
        raise ValueError("a name is 2 to 30 of: a-z 0-9 _ . -")
    password = password or secrets.token_urlsafe(9)
    if len(password) < 8:
        raise ValueError("a password has at least 8 characters")
    salt = secrets.token_hex(16)
    with LOCK:
        accounts = read()
        accounts[name] = {"salt": salt, "hash": hashed(password, salt), "rounds": ROUNDS,
                          "made": accounts.get(name, {}).get("made") or time.strftime("%Y-%m-%d")}
        write(accounts)
    log_out_all(name)
    return password


def remove(name):
    with LOCK:
        accounts = read()
        if accounts.pop(name, None) is None:
            raise ValueError(f"no account {name}")
        write(accounts)
    log_out_all(name)


def right(name, password):
    """Is this the account's password? A name that is not there costs the same time as a wrong password."""
    account = read().get(name) or {"salt": "00", "hash": "", "rounds": ROUNDS}
    return hmac.compare_digest(hashed(password, account["salt"], account.get("rounds", ROUNDS)), account["hash"])


def log_in(name, password):
    """The token of a new login (the page's cookie), or ValueError with what to show."""
    name = str(name or "").strip().lower()
    with LOCK:
        count, last = WRONG.get(name, (0, 0))
        if count >= TRIES and time.time() - last < LOCKED:
            raise ValueError("too many wrong passwords: wait ten minutes")
    if not right(name, str(password or "")):
        with LOCK:
            count, last = WRONG.get(name, (0, 0))
            WRONG[name] = (1 if time.time() - last > LOCKED else count + 1, time.time())
            if len(WRONG) > 10_000:  # names nobody has: not kept forever
                WRONG.clear()
        raise ValueError("wrong name or password")
    token = secrets.token_urlsafe(32)
    with LOCK:
        WRONG.pop(name, None)
        for old in [t for t, (_, until) in SESSIONS.items() if until < time.time()]:
            del SESSIONS[old]
        SESSIONS[token] = (name, time.time() + KEPT)
    return token


def who(token):
    """The name that is logged in with this token, or None."""
    with LOCK:
        name, until = SESSIONS.get(token or "", (None, 0))
    if not name or until < time.time() or name not in read():  # a removed account is out at once
        return None
    return name


def log_out(token):
    with LOCK:
        SESSIONS.pop(token or "", None)


def log_out_all(name):
    with LOCK:
        for token in [t for t, (n, _) in SESSIONS.items() if n == name]:
            del SESSIONS[token]


def main():
    parser = argparse.ArgumentParser(description="The accounts of the web page (flow_page.py --accounts)")
    parser.add_argument("what", choices=["add", "remove", "list"])
    parser.add_argument("name", nargs="?")
    args = parser.parse_args()
    try:
        if args.what == "list":
            for name, account in sorted(read().items()):
                print(f"{name:32}made {account.get('made', '?')}")
            if not read():
                print("no accounts yet:  python scripts/pipeline/accounts.py add <name>")
        elif not args.name:
            parser.error("which name?")
        elif args.what == "add":
            new = args.name.strip().lower() not in read()
            password = add(args.name)
            print(f"{'account' if new else 'new password for'} {args.name.strip().lower()}\npassword: {password}\n"
                  "It is shown only this once. Give it to that person; a forgotten one: add the name again.")
        else:
            remove(args.name)
            print(f"removed {args.name}")
    except ValueError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()

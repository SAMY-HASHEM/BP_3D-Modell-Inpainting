"""Tests of the web page's back end (scripts/pipeline/flow_page.py) and of the edit node, without GPU, server or browser.

    python -W ignore::ResourceWarning -m unittest discover tests      # any Python 3, from the project folder

Everything is written into a temporary folder: the project's own flows/ and pictures/ are not touched.
"""
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "scripts", "pipeline"))

import accounts  # noqa: E402
import flow as flows  # noqa: E402
import flow_page as page  # noqa: E402
import nodes_core as core  # noqa: E402


class InTemporaryProject(unittest.TestCase):
    """flow_page with its folders in a temporary project."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = os.path.realpath(self.temporary.name)
        self.root = root
        patches = [mock.patch.object(core, "PROJECT", root), mock.patch.object(core, "OUTPUTS", os.path.join(root, "outputs")),
                   mock.patch.object(page, "FLOWS", os.path.join(root, "flows")),
                   mock.patch.object(page, "HAND", os.path.join(root, "flows", "hand")),
                   mock.patch.object(page, "UPLOADS", os.path.join(root, "pictures", "uploads")),
                   mock.patch.object(page, "GPU_HERE", True),  # a computer with a GPU; WithoutAGpu tests the other one
                   mock.patch.object(page, "vision_ai_here", lambda: True)]  # nothing is asked: the vision AI counts as there
        for one in patches:
            one.start()
            self.addCleanup(one.stop)
        self.addCleanup(self.temporary.cleanup)
        os.makedirs(page.FLOWS)

    def upload(self, flow, role, name, body=b"\x89PNG picture"):
        """As the page's POST /api/upload: gives the path from the project folder."""
        import io
        return page.save_upload(f"/api/upload?flow={flow}&role={role}&name={name}", io.BytesIO(body), len(body))

    def flow(self, name):
        return json.load(open(os.path.join(page.FLOWS, name + ".json")))


class Uploads(InTemporaryProject):
    def test_a_picture_is_saved_under_the_project_and_role(self):
        path = self.upload("demo", "object", "front.png")
        self.assertEqual(path, "pictures/uploads/demo/object/front.png")
        self.assertTrue(os.path.isfile(os.path.join(self.root, path)))

    def test_a_name_cannot_leave_the_upload_folder(self):
        path = self.upload("demo", "part", "..%2F..%2F..%2Fevil.png")
        self.assertTrue(path.startswith("pictures/uploads/demo/part/"))
        self.assertNotIn("..", path)

    def test_only_pictures_and_videos(self):  # and a .glb of the object (the run starts from it), never of the part
        for role, name in (("object", "script.py"), ("object", "noending"), ("part", "model.glb")):
            with self.assertRaises(ValueError):
                self.upload("demo", role, name)
        self.assertEqual(self.upload("demo", "object", "model.glb"), "pictures/uploads/demo/object/model.glb")

    def test_project_name_and_role_are_checked(self):
        with self.assertRaises(ValueError):
            self.upload("../demo", "object", "a.png")
        with self.assertRaises(ValueError):
            self.upload("demo", "other", "a.png")


class CreatingAFlow(InTemporaryProject):
    def create(self, name="demo", parts=0, **more):
        items = [{"path": self.upload(name, "object", "front.png"), "role": "object", "prompt": "the  front "}]
        items += [{"path": self.upload(name, "part", f"part{k}.png"), "role": "part", "prompt": ""} for k in range(parts)]
        page.create_flow({"flow": name, "prompt": "a test", "items": items, **more})
        return self.flow(name)

    def test_only_an_object_ends_at_the_base_model(self):
        made = self.create(settings={"seed": 7, "steps": 12, "resolution": "512", "where": "pc"})
        self.assertEqual([n["id"] for n in made["nodes"]], ["object", "object_sheet", "base"])
        base = made["nodes"][-1]
        self.assertEqual((base["seed"], base["steps"], base["resolution"], base["where"]), (7, 12, "512", "pc"))
        self.assertEqual(made["nodes"][0]["prompts"], ["front.png: the front"])
        self.assertEqual(made["nodes"][1]["what"], "the front")  # the row's prompt is what the object is called
        self.assertEqual(made["made_by"], "app")

    def test_a_part_gives_the_structure_of_the_first_success(self):
        # docs/FIRST-SUCCESS.md: the part as its own model, placed in 3D, the plan judged, details and colours only
        made = self.create(parts=1)
        ids = [n["id"] for n in made["nodes"]]
        self.assertEqual(ids, ["object", "part", "object_sheet", "part_sheet", "base", "part_views", "edit", "part_model", "place",
                               "check", "details", "judge_details", "colours", "judge_colours", "finish"])
        self.assertEqual(node_types := {n["id"]: n["type"] for n in made["nodes"]}["colours"], "part_colours")  # not TRELLIS.2's
        node = {n["id"]: n for n in made["nodes"]}
        # one picture of the part: the picture AI makes its other sides first (its back is not guessed by TRELLIS.2)
        self.assertEqual(node["part_views"]["in"], {"data": "part_sheet"})
        self.assertEqual(node["edit"]["in"], {"model": "base", "part": "part_views", "object": "object_sheet"})
        self.assertEqual(node["part_model"]["in"], {"data": "part_views"})
        self.assertEqual(node["place"]["in"], {"part_model": "part_model", "edit": "edit"})
        self.assertEqual(node["details"]["in"], {"model": "place", "edit": "check"})  # waits for the plan's pass
        self.assertEqual(node["finish"]["in"], {"model": "judge_colours", "edit": "check"})
        edit = node["edit"]
        self.assertNotIn("prompt", edit)
        self.assertNotIn("what", next(n for n in made["nodes"] if n["id"] == "part_sheet"))  # no prompt: the local AI says what it is

    def test_an_upload_folder_linked_elsewhere_keeps_its_files(self):
        # /home full (Nov 19): pictures/uploads/<flow> linked to /tmp; its files were deleted as "taken off the page"
        elsewhere = tempfile.mkdtemp()
        os.makedirs(page.UPLOADS, exist_ok=True)
        os.symlink(elsewhere, os.path.join(page.UPLOADS, "demo"))
        made = self.create()
        self.assertTrue(os.path.isfile(os.path.join(elsewhere, "object", "front.png")))
        self.assertEqual(made["nodes"][0]["files"], ["pictures/uploads/demo/object/front.png"])

    def test_a_part_with_several_pictures_gets_no_made_sides(self):
        made = self.create(parts=2)
        node = {n["id"]: n for n in made["nodes"]}
        self.assertNotIn("part_views", node)
        self.assertEqual(node["part_model"]["in"], {"data": "part_sheet"})
        self.assertEqual(node["edit"]["in"]["part"], "part_sheet")

    def test_a_part_from_words_needs_no_part_input(self):
        # Nov 19: the painted part is cut out and gets its own model, as a part with pictures (from noise: a thin ring)
        made = self.create(part_prompt="  a wide   pointed hat on the head ")
        ids = [n["id"] for n in made["nodes"]]
        self.assertNotIn("part", ids)
        self.assertEqual(ids, ["object", "object_sheet", "base", "edit", "cut_part", "part_sheet", "part_views", "part_model",
                               "place", "check", "details", "judge_details", "colours", "judge_colours", "finish"])
        node = {n["id"]: n for n in made["nodes"]}
        self.assertEqual(node["edit"]["prompt"], "a wide pointed hat on the head")
        self.assertEqual(node["edit"]["in"], {"model": "base", "object": "object_sheet"})
        self.assertEqual(node["part_sheet"]["in"], {"pictures": "cut_part"})
        self.assertEqual(node["part_sheet"]["what"], "a wide pointed hat on the head")

    def test_a_part_from_words_can_still_grow_from_noise(self):
        made = self.create(part_prompt="a hat", from_noise=True)
        ids = [n["id"] for n in made["nodes"]]
        self.assertNotIn("cut_part", ids)
        self.assertEqual(ids[-1], "judge_colours")

    def test_words_and_pictures_of_a_part_together_are_refused(self):
        with self.assertRaises(ValueError):
            self.create(parts=1, part_prompt="a hat")

    def test_every_node_lists_only_nodes_before_it(self):
        for made in (self.create("one", parts=2), self.create("two", part_prompt="a hat")):
            seen = set()
            for node in made["nodes"]:
                for sources in flows.links(node).values():
                    self.assertTrue(set(sources) <= seen, node["id"])
                self.assertIn(node["type"], flows.TYPES)
                seen.add(node["id"])

    def test_it_needs_an_object(self):
        item = {"path": self.upload("demo", "part", "crown.png"), "role": "part"}
        with self.assertRaises(ValueError):
            page.create_flow({"flow": "demo", "items": [item]})

    def test_a_file_from_elsewhere_is_refused(self):
        other = os.path.join(self.root, "secret.png")
        open(other, "w").close()
        for path in ("secret.png", "../secret.png", self.upload("other", "object", "a.png")):
            with self.assertRaises(ValueError):
                page.create_flow({"flow": "demo", "items": [{"path": path, "role": "object"}]})

    def test_a_flow_made_by_hand_is_never_overwritten(self):
        page.save_flow(os.path.join(page.FLOWS, "mine.json"), {"name": "mine", "nodes": []})
        with self.assertRaises(ValueError):
            self.create("mine")
        self.create("mine2")
        self.create("mine2")  # its own flows it may write again

    def test_settings_that_do_not_fit(self):
        for settings in ({"resolution": "2048"}, {"where": "moon"}, {"steps": 0}, {"steps": "many"}, {"seed": -1}):
            with self.assertRaises(ValueError, msg=settings):
                self.create(settings=settings)

    def test_what_was_taken_off_the_page_leaves_the_upload_folder(self):
        self.create(parts=2)
        folder = os.path.join(page.UPLOADS, "demo", "part")
        self.assertEqual(len(os.listdir(folder)), 2)
        self.create(parts=0)
        self.assertEqual(os.listdir(folder), [])


class WithoutAGpu(InTemporaryProject):
    """A laptop (no NVIDIA GPU): every run goes to the uni server, whatever the page asks for."""

    def setUp(self):
        super().setUp()
        patch = mock.patch.object(page, "GPU_HERE", False)
        patch.start()
        self.addCleanup(patch.stop)

    def test_a_new_flow_runs_on_the_server_whatever_was_chosen(self):
        for where in ("pc", "server", "all"):
            item = {"path": self.upload("demo", "object", "front.png"), "role": "object"}
            page.create_flow({"flow": "demo", "items": [item], "settings": {"where": where}})
            self.assertEqual(self.flow("demo").get("runs"), "server", where)

    def test_run_sends_any_flow_to_the_server(self):
        page.save_flow(os.path.join(page.FLOWS, "mine.json"), {"name": "mine", "nodes": [{"id": "a", "type": "input", "files": []}]})
        with mock.patch.object(page, "start_remote") as remote, mock.patch.object(page.subprocess, "Popen") as local:
            page.start_run("mine", upto="a")
        remote.assert_called_once_with("mine")
        local.assert_not_called()

    def test_no_local_ai(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OLLAMA_HOST", None)
            with self.assertRaises(ValueError):
                page.understand({"prompt": "a hat", "items": [{"name": "a.png"}]})


class YourOwnModel(InTemporaryProject):
    """A .glb of the object: the run starts from it (the base node takes it), its picture still makes the data sheet."""

    def test_the_model_is_the_base_and_the_picture_the_sheet(self):
        model = self.upload("demo", "object", "robot.glb", b"glTF model")
        front = self.upload("demo", "object", "front.png")
        sword = self.upload("demo", "part", "sword.png")
        page.create_flow({"flow": "demo", "items": [{"path": model, "role": "object"}, {"path": front, "role": "object"},
                                                     {"path": sword, "role": "part"}]})
        nodes = {n["id"]: n for n in self.flow("demo")["nodes"]}
        self.assertEqual(nodes["base"]["given"], model)
        self.assertEqual(nodes["object"]["files"], [front])
        self.assertTrue(os.path.isfile(os.path.join(self.root, model)))  # kept in the upload folder

    def test_a_model_needs_a_picture_and_is_never_the_part(self):
        model = self.upload("demo", "object", "robot.glb", b"glTF model")
        with self.assertRaises(ValueError):
            page.create_flow({"flow": "demo", "items": [{"path": model, "role": "object"}]})
        with self.assertRaises(ValueError):
            self.upload("demo", "part", "sword.glb", b"glTF model")

    def test_the_base_node_takes_the_given_model(self):
        glb = os.path.join(self.root, "pictures", "uploads", "demo", "object", "robot.glb")
        os.makedirs(os.path.dirname(glb), exist_ok=True)
        open(glb, "wb").write(b"glTF model")
        job = core.base_model({"name": "demo", "path": ""}, given="pictures/uploads/demo/object/robot.glb")
        self.assertEqual(job["steps"], [])
        self.assertEqual(os.path.normpath(job["outputs"]["model"]["glb"]), os.path.normpath(glb))


class OnTheServer(InTemporaryProject):
    """flow_page.py --on-server: the page runs on the uni server itself, so every node runs there, nothing over Sciebo."""

    def test_every_node_runs_here_whatever_was_chosen(self):
        with mock.patch.object(page, "ON_SERVER", True):
            for where in ("all", "server", "pc"):
                item = {"path": self.upload("demo", "object", "front.png"), "role": "object"}
                page.create_flow({"flow": "demo", "items": [item], "settings": {"where": where}})
                flow = self.flow("demo")
                self.assertNotIn("runs", flow, where)
                self.assertEqual(next(n for n in flow["nodes"] if n["type"] == "base")["where"], "pc", where)
                part = self.upload("demo", "part", "crown.png")
                page.create_flow({"flow": "demo", "items": [item, {"path": part, "role": "part"}], "settings": {"where": where}})
                self.assertEqual({n["where"] for n in self.flow("demo")["nodes"] if n["type"] == "inpaint"}, {"pc"}, where)  # the forge too


class TheBridge(InTemporaryProject):
    """The Sciebo login of the bridge: tried before it is saved, its password never sent back."""

    def setUp(self):
        super().setUp()
        self.cloud = page.cloud_module()
        self.config = os.path.join(self.root, "cloud.json")
        for one in (mock.patch.object(self.cloud, "CONFIG", self.config), mock.patch.dict(page.BRIDGE, {"state": "idle"})):
            one.start()
            self.addCleanup(one.stop)

    def test_no_login_yet(self):
        self.assertIsNone(page.cloud_login()["login"])
        with self.assertRaises(ValueError):
            page.test_cloud()

    def test_a_login_that_works_is_saved_without_showing_its_password(self):
        with mock.patch.object(self.cloud.Cloud, "make_folder"), mock.patch.object(self.cloud.Cloud, "list", return_value=["a.zip"]):
            page.save_cloud_login({"user": "anna", "password": "secret-app-password"})
            self.assertEqual(page.test_cloud(), 1)
        self.assertEqual(json.load(open(self.config))["password"], "secret-app-password")
        shown = json.dumps(page.cloud_login())
        self.assertIn("anna", shown)
        self.assertNotIn("secret-app-password", shown)

    def test_a_refused_login_is_not_saved(self):
        refused = urllib.error.HTTPError("https://sciebo", 401, "Unauthorized", {}, None)
        with mock.patch.object(self.cloud.Cloud, "make_folder", side_effect=refused):
            with self.assertRaises(ValueError) as said:
                page.save_cloud_login({"user": "anna", "password": "wrong"})
        self.assertIn("refused", str(said.exception))
        self.assertFalse(os.path.exists(self.config))

    def test_both_are_needed(self):
        for wanted in ({"user": "anna"}, {"password": "x"}, {"user": " ", "password": "x"}):
            with self.assertRaises(ValueError):
                page.save_cloud_login(wanted)


class NodeSettings(InTemporaryProject):
    def setUp(self):
        super().setUp()
        page.save_flow(os.path.join(page.FLOWS, "demo.json"), {"name": "demo", "nodes": [
            {"id": "base", "type": "base", "seed": 42}, {"id": "edit", "type": "edit", "in": {"model": "base"}},
            {"id": "shape", "type": "inpaint", "seeds": [1]}]})

    def node(self, key):
        return next(n for n in self.flow("demo")["nodes"] if n["id"] == key)

    def test_each_setting_keeps_the_kind_of_its_default(self):
        page.set_settings("demo", {"node": "base", "settings": {"seed": "7", "steps": 30.0, "resolution": 512}})
        self.assertEqual((self.node("base")["seed"], self.node("base")["steps"], self.node("base")["resolution"]), (7, 30, "512"))
        page.set_settings("demo", {"node": "shape", "settings": {"seeds": "1, 2;3"}})
        self.assertEqual(self.node("shape")["seeds"], [1, 2, 3])
        page.set_settings("demo", {"node": "edit", "settings": {"replace": 1, "width": "0.5", "prompt": " a hat ", "prompt_seed": "3"}})
        edit = self.node("edit")
        self.assertEqual((edit["replace"], edit["width"], edit["prompt"], edit["prompt_seed"]), (True, 0.5, "a hat", 3))

    def test_a_default_that_was_not_in_the_file_stays_out(self):
        page.set_settings("demo", {"node": "base", "settings": {"steps": 25}})
        self.assertNotIn("steps", self.node("base"))

    def test_what_cannot_be_set(self):
        for node, settings in (("base", {"faces": 1}), ("base", {"where": "moon"}), ("base", {"seed": "x"}), ("shape", {"seeds": ""}),
                               ("nobody", {"seed": 1})):
            with self.assertRaises(ValueError, msg=(node, settings)):
                page.set_settings("demo", {"node": node, "settings": settings})


class TheEditNode(InTemporaryProject):
    """nodes_core.edit: the folder's name is the node's memory, so it must not change for flows that are there already."""

    def setUp(self):
        super().setUp()
        self.model = {"glb": os.path.join(self.root, "model.glb"), "name": "demo"}
        for name, lists in (("object", {"base": ["front"]}), ("part", {"sides": ["side"]})):
            json.dump({"what": f"the {name}", "views": [{"name": "front", "path": "front.png"}, {"name": "side", "path": "side.png"}],
                       "lists": lists}, open(os.path.join(self.root, name + ".json"), "w"))
        self.object = {"path": os.path.join(self.root, "object.json"), "name": "demo"}
        self.part = {"path": os.path.join(self.root, "part.json"), "name": "demo"}
        open(os.path.join(self.root, "side.png"), "w").close()

    def test_with_a_part_picture_nothing_changed(self):
        job = core.edit(self.model, self.part, self.object)
        settings = json.load(open(os.path.join(job["folder"], "settings.json")))
        self.assertEqual(sorted(settings), ["fit", "model", "reference", "replace", "sides", "sink", "width"])  # as before the prompt
        command = job["steps"][0][0]
        self.assertTrue(command[3].endswith("part_sides"))
        self.assertNotIn("--prompt", command)
        self.assertEqual(job["outputs"]["edit"]["part"], "the part")

    def test_a_prompt_beside_a_part_picture_is_not_used(self):
        self.assertEqual(core.edit(self.model, self.part, self.object)["folder"],
                         core.edit(self.model, self.part, self.object, prompt="a hat")["folder"])

    def test_a_part_picture_placed_by_words(self):
        job = core.edit(self.model, self.part, self.object, at=" in the  left hand ")
        command = job["steps"][0][0]
        self.assertTrue(command[3].endswith("part_sides"))
        self.assertEqual(command[command.index("--prompt") + 1], "the part in the left hand")
        self.assertEqual(json.load(open(os.path.join(job["folder"], "settings.json")))["at"], "in the left hand")
        self.assertNotEqual(job["folder"], core.edit(self.model, self.part, self.object)["folder"])
        by_hand = core.edit(self.model, self.part, self.object, at="in the left hand", place=[1, 2, 3])["steps"][0][0]
        self.assertNotIn("--prompt", by_hand)  # a place by hand wins

    def test_words_only(self):
        job = core.edit(self.model, None, self.object, prompt=" a wide  hat ", prompt_seed=2)
        command = job["steps"][0][0]
        at = command.index("--prompt")
        self.assertEqual(command[at:at + 4], ["--prompt", "a wide hat", "--prompt-seed", "2"])
        self.assertEqual(job["outputs"]["edit"]["part"], "a wide hat")
        other = core.edit(self.model, None, self.object, prompt="a wide hat", prompt_seed=3)
        self.assertNotEqual(job["folder"], other["folder"])  # another seed is another result

    def test_neither_a_part_nor_words(self):
        with self.assertRaises(ValueError):
            core.edit(self.model, None, self.object)


class TheDataSheetNode(InTemporaryProject):
    def test_the_users_words_are_only_in_the_folders_name_when_given(self):
        picture = os.path.join(self.root, "a.png")
        open(picture, "w").close()
        pics = {"name": "demo", "files": [picture]}
        plain, named = core.data_sheet(pics), core.data_sheet(pics, what=" the  girl's head ")
        self.assertEqual(sorted(json.load(open(os.path.join(plain["folder"], "settings.json")))), ["ai", "files", "frames", "role"])
        self.assertNotEqual(plain["folder"], named["folder"])
        self.assertEqual(named["steps"][0][0][-2:], ["--what", "the girl's head"])
        self.assertNotIn("--what", plain["steps"][0][0])


class OneSetOfDefaults(unittest.TestCase):
    """A setting's default is written down in the page's back end (SETTINGS) and in the node itself (nodes_core):
    the page shows the first, the flow runs with the second. They must say the same."""

    def test_the_pages_defaults_are_the_nodes_defaults(self):
        import inspect
        nodes = {"data_sheet": core.data_sheet, "base": core.base_model, "edit": core.edit, "inpaint": core.inpaint_step,
                 "judge": core.edit_judge, "place": core.place, "check": core.plan_check}
        same = lambda value: list(value) if isinstance(value, (list, tuple)) else "" if value is None else value
        for kind, function in nodes.items():
            real = {name: p.default for name, p in inspect.signature(function).parameters.items() if p.default is not p.empty}
            for key, shown in page.SETTINGS[kind].items():
                self.assertIn(key, real, f"{kind}: {key} is on the page but the node has no such setting")
                self.assertEqual(same(shown), same(real[key]), f"{kind}: {key}")

    def test_every_node_type_is_known_everywhere(self):
        self.assertEqual(set(page.SETTINGS), set(flows.TYPES))
        self.assertEqual(set(flows.COLOURS), set(flows.TYPES))


class AJudgeThatSaysRedo(unittest.TestCase):
    """A judge's redo stops the flow there (Nov 19: details judged redo, and the colours step started on it anyway)."""

    def flow_with(self, verdict, **judge):
        made = flows.Flow.__new__(flows.Flow)
        made.nodes = {"judge_details": {"id": "judge_details", "type": "judge", **judge}, "edit": {"id": "edit", "type": "edit"}}
        made.state = {"judge_details": {"outputs": {"verdict": verdict}}, "edit": {"outputs": {"verdict": "redo"}}}
        return made

    def test_redo_stops_it(self):
        self.assertIn("redo (part 46", self.flow_with("redo (part 46 is below 70)").rejected("judge_details"))

    def test_pass_goes_on(self):
        self.assertEqual(self.flow_with("pass").rejected("judge_details"), "")

    def test_it_can_be_let_go_on(self):
        self.assertEqual(self.flow_with("redo", stop_on_redo=False).rejected("judge_details"), "")

    def test_only_a_judge_stops_it(self):
        self.assertEqual(self.flow_with("pass").rejected("edit"), "")


class AProxyThatRefuses(unittest.TestCase):
    """The uni server's proxy is sometimes down: a run then goes without it, with the models from disk."""

    def test_a_dead_proxy_is_left_out(self):
        import socket
        free = socket.socket()
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
        free.close()  # nothing listens there now
        env = {"https_proxy": f"http://127.0.0.1:{port}", "http_proxy": f"http://127.0.0.1:{port}", "no_proxy": "git.uni-due.de,"}
        page.without_dead_proxy(env)
        self.assertNotIn("https_proxy", env)
        self.assertNotIn("http_proxy", env)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["no_proxy"], "git.uni-due.de,127.0.0.1,localhost")

    def test_a_proxy_that_answers_stays(self):
        import socket
        listening = socket.socket()
        listening.bind(("127.0.0.1", 0))
        listening.listen(1)
        self.addCleanup(listening.close)
        env = {"https_proxy": f"http://127.0.0.1:{listening.getsockname()[1]}"}
        page.without_dead_proxy(env)
        self.assertIn("https_proxy", env)
        self.assertNotIn("HF_HUB_OFFLINE", env)

    def test_without_a_proxy_only_this_computer_is_named(self):
        env = {}
        page.without_dead_proxy(env)
        self.assertEqual(env, {"no_proxy": "127.0.0.1,localhost", "NO_PROXY": "127.0.0.1,localhost"})


class WithServer(InTemporaryProject):
    """The page's server on a free port; ask() is a browser's request."""

    def setUp(self):
        super().setUp()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), page.Page)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.address = f"http://127.0.0.1:{self.server.server_port}"

    def ask(self, path, body=None, headers=None):
        request = urllib.request.Request(self.address + path, data=body, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=10) as answer:
                return answer.status, answer.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()


class OverHttp(WithServer):
    def test_the_pages_and_the_flow_list(self):
        self.assertEqual(self.ask("/app")[0], 200)
        self.assertEqual(self.ask("/")[0], 200)
        self.assertEqual(json.loads(self.ask("/api/flows")[1]), [])

    def test_a_post_without_the_pages_header_is_refused(self):
        body = json.dumps({"flow": "demo"}).encode()
        self.assertEqual(self.ask("/api/run", body, {"Content-Type": "application/json"})[0], 403)
        self.assertEqual(self.ask("/api/run", body, {"X-Flow-Page": "1", "Origin": "http://evil.example"})[0], 403)

    def test_only_pictures_out_of_the_open_folders(self):
        os.makedirs(os.path.join(self.root, "pictures"))
        open(os.path.join(self.root, "pictures", "a.png"), "wb").write(b"x")
        open(os.path.join(self.root, "secret.png"), "wb").write(b"x")
        open(os.path.join(self.root, "pictures", "notes.txt"), "wb").write(b"x")
        self.assertEqual(self.ask("/file/pictures/a.png")[0], 200)
        for path in ("/file/secret.png", "/file/pictures/../secret.png", "/file/pictures/notes.txt", "/file/pictures/%2e%2e/secret.png"):
            self.assertEqual(self.ask(path)[0], 404, path)

    def test_a_folder_linked_into_outputs_is_shown(self):
        # /home full: a run's work folder moved elsewhere and linked back (Nov 19)
        elsewhere = tempfile.mkdtemp()
        open(os.path.join(elsewhere, "b.png"), "wb").write(b"x")
        os.makedirs(os.path.join(self.root, "outputs"), exist_ok=True)
        os.symlink(elsewhere, os.path.join(self.root, "outputs", "moved"))
        self.assertEqual(self.ask("/file/outputs/moved/b.png")[0], 200)
        self.assertEqual(self.ask("/file/outputs/moved/../../secret.png")[0], 404)

    def test_upload_then_create_as_the_page_does(self):
        mine = {"X-Flow-Page": "1", "Content-Type": "application/octet-stream"}
        code, answer = self.ask("/api/upload?flow=demo&role=object&name=front.png", b"\x89PNG", mine)
        self.assertEqual(code, 200)
        wanted = {"flow": "demo", "items": [{"path": json.loads(answer)["path"], "role": "object"}], "part_prompt": "a hat"}
        code, answer = self.ask("/api/app/create", json.dumps(wanted).encode(), {"X-Flow-Page": "1", "Content-Type": "application/json"})
        self.assertEqual((code, json.loads(answer)), (200, {"ok": True}))
        listed = json.loads(self.ask("/api/flows")[1])
        self.assertEqual([(f["file"], f["nodes"], f["app"]) for f in listed], [("demo", 15, True)])
        code, answer = self.ask("/api/upload?flow=demo&role=part&name=model.glb", b"x", mine)
        self.assertEqual(code, 400)
        self.assertIn("no picture or video", json.loads(answer)["error"])


class Accounts(WithServer):
    """flow_page.py --accounts: nothing without a login, the accounts are handed out (accounts.py)."""
    mine = {"X-Flow-Page": "1", "Content-Type": "application/json"}

    def setUp(self):
        super().setUp()
        for one in (mock.patch.object(accounts, "FILE", os.path.join(self.root, "accounts.json")), mock.patch.object(accounts, "ROUNDS", 1000),
                    mock.patch.object(accounts, "SESSIONS", {}), mock.patch.object(accounts, "WRONG", {}), mock.patch.object(page.Page, "accounts", True)):
            one.start()
            self.addCleanup(one.stop)
        self.password = accounts.add("anna")

    def log_in(self, name="anna", password=None):
        """As the login page: gives the answer's code and the cookie to send from then on."""
        body = json.dumps({"name": name, "password": password or self.password}).encode()
        request = urllib.request.Request(self.address + "/api/login", data=body, headers=self.mine)
        try:
            with urllib.request.urlopen(request, timeout=10) as answer:
                return answer.status, answer.headers["Set-Cookie"].split(";")[0]
        except urllib.error.HTTPError as error:
            return error.code, None

    def test_the_file_keeps_no_password(self):
        kept = open(accounts.FILE).read()
        self.assertNotIn(self.password, kept)
        self.assertTrue(accounts.right("anna", self.password))
        self.assertFalse(accounts.right("anna", self.password + "x"))
        self.assertFalse(accounts.right("nobody", self.password))
        for name in ("a", "Anna Maria", "../anna", ""):
            with self.assertRaises(ValueError, msg=name):
                accounts.add(name)

    def test_nothing_without_a_login(self):
        self.assertEqual(self.ask("/api/flows")[0], 401)
        self.assertEqual(self.ask("/file/pictures/a.png")[0], 401)
        self.assertEqual(self.ask("/api/run", b'{"flow": "demo"}', self.mine)[0], 401)
        self.assertEqual(self.ask("/api/upload?flow=demo&role=object&name=a.png", b"x", {"X-Flow-Page": "1"})[0], 401)
        code, body = self.ask("/app")  # the pages lead to the login page
        self.assertEqual(code, 200)
        self.assertIn(b"Log in", body)

    def test_log_in_and_out(self):
        self.assertEqual(self.log_in(password="wrong password")[0], 401)
        self.assertEqual(self.log_in(name="nobody")[0], 401)
        code, cookie = self.log_in()
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(self.ask("/api/flows", headers={"Cookie": cookie})[1]), [])
        self.assertEqual(json.loads(self.ask("/api/me", headers={"Cookie": cookie})[1]), {"accounts": True, "name": "anna", "gpu": True, "on_server": False})
        self.assertEqual(self.ask("/api/blender", b'{"model": "a.glb"}', {**self.mine, "Cookie": cookie})[0], 400)
        self.assertEqual(self.ask("/api/logout", b"{}", {**self.mine, "Cookie": cookie})[0], 200)
        self.assertEqual(self.ask("/api/flows", headers={"Cookie": cookie})[0], 401)

    def test_a_removed_account_and_a_new_password_end_the_login(self):
        _, cookie = self.log_in()
        new = accounts.add("anna")
        self.assertEqual(self.ask("/api/flows", headers={"Cookie": cookie})[0], 401)
        self.assertEqual(self.log_in()[0], 401)  # the old password
        _, cookie = self.log_in(password=new)
        accounts.remove("anna")
        self.assertEqual(self.ask("/api/flows", headers={"Cookie": cookie})[0], 401)

    def test_many_wrong_passwords_lock_the_name(self):
        for _ in range(accounts.TRIES):
            self.assertEqual(self.log_in(password="wrong password")[0], 401)
        self.assertEqual(self.log_in()[0], 401)  # even the right one now


class WithoutAccounts(WithServer):
    def test_nobody_is_asked(self):
        self.assertEqual(json.loads(self.ask("/api/me")[1]), {"accounts": False, "name": None, "gpu": True, "on_server": False})
        self.assertEqual(self.ask("/login")[0], 404)
        self.assertEqual(self.ask("/api/login", b"{}", {"X-Flow-Page": "1"})[0], 404)


if __name__ == "__main__":
    unittest.main()

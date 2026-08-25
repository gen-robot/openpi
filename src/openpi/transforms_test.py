import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_mask_image_upper_half():
    left_image = np.full((5, 4, 3), 255, dtype=np.uint8)
    right_image = np.full((5, 4, 3), 127, dtype=np.uint8)
    data = {"image": {"left_wrist_0_rgb": left_image, "right_wrist_0_rgb": right_image}}

    transformed = _transforms.MaskImageUpperHalf("left_wrist_0_rgb")(data)

    assert np.all(transformed["image"]["left_wrist_0_rgb"][:2] == 0)
    assert np.all(transformed["image"]["left_wrist_0_rgb"][2:] == 255)
    assert np.all(transformed["image"]["right_wrist_0_rgb"] == 127)
    assert np.all(left_image == 255)  # The source array is not mutated.


def test_mask_image_upper_half_supports_leading_dimensions():
    image = np.ones((2, 6, 4, 3), dtype=np.float32)

    transformed = _transforms.MaskImageUpperHalf("left")({"image": {"left": image}})

    assert np.all(transformed["image"]["left"][:, :3] == 0)
    assert np.all(transformed["image"]["left"][:, 3:] == 1)


def test_mask_image_upper_half_probability(monkeypatch):
    image = np.ones((6, 4, 3), dtype=np.float32)
    transform = _transforms.MaskImageUpperHalf("left", probability=0.25)

    monkeypatch.setattr(_transforms.np.random, "random", lambda: 0.24)
    masked = transform({"image": {"left": image}})
    assert np.all(masked["image"]["left"][:3] == 0)

    monkeypatch.setattr(_transforms.np.random, "random", lambda: 0.25)
    unmasked = transform({"image": {"left": image}})
    assert unmasked["image"]["left"] is image


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_mask_image_upper_half_rejects_invalid_probability(probability):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _transforms.MaskImageUpperHalf("left", probability=probability)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})

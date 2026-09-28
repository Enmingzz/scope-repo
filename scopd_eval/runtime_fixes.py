"""Existing evaluator fixes, applied to pinned functions in this process only."""
import ast
import inspect
import textwrap


def patch_wrapper_source(source, video_timing=False):
    tree = ast.parse(textwrap.dedent(source))
    tree.body[0].decorator_list = []
    counts = {'fps_return': 0, 'fps_forward': 0, 'counters': 0}
    for node in ast.walk(tree):
        if (video_timing and isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == 'process_vision_info'):
            assert ast.unparse(node.targets[0]) == '(images, videos)'
            node.targets[0].elts.append(ast.Name(id='video_kwargs', ctx=ast.Store()))
            node.value.keywords.append(ast.keyword(arg='return_video_kwargs', value=ast.Constant(True)))
            counts['fps_return'] += 1
        if (video_timing and isinstance(node, ast.Call) and ast.unparse(node.func) == 'self.processor'
                and {kw.arg for kw in node.keywords} == {'text', 'images', 'videos', 'padding', 'return_tensors'}):
            node.keywords.append(ast.keyword(arg=None, value=ast.parse(
                'video_kwargs if videos is not None else {}', mode='eval').body))
            counts['fps_forward'] += 1
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'sample_initial_visual_tokens':
            for grid in ('image_grid_thw', 'video_grid_thw'):
                if ast.unparse(node.value) == f'inputs.{grid}.prod().item() // 4':
                    node.value = ast.parse(f'inputs.{grid}.prod(dim=-1).sum().item() // 4', mode='eval').body
                    counts['counters'] += 1
    assert counts == {'fps_return': int(video_timing), 'fps_forward': int(video_timing), 'counters': 2}, counts
    return ast.fix_missing_locations(tree)


def install_wrapper(video_timing=False):
    from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat
    if getattr(Qwen2VLChat, '_portable_runtime_fixes', False):
        return
    function = Qwen2VLChat.generate_inner_transformers
    if function.__code__.co_freevars:
        raise RuntimeError('Unexpected closure in patched wrapper')
    tree = patch_wrapper_source(inspect.getsource(function), video_timing)
    namespace = {}
    exec(compile(tree, __file__, 'exec'), function.__globals__, namespace)
    function.__code__ = namespace[function.__name__].__code__
    Qwen2VLChat._portable_runtime_fixes = True

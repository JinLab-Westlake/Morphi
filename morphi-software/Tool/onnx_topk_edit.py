# tools/onnx_topk_edit.py
import onnx, numpy as np
from onnx import helper, numpy_helper, TensorProto

PASSTHRU = {'Reshape','Identity','Unsqueeze','Cast','Squeeze'}

def load(p): return onnx.load(p)
def save(m,p): onnx.save(m,p)

def maps(m):
    g=m.graph
    out2node={o:n for n in g.node for o in n.output}
    init={i.name:i for i in g.initializer}
    return out2node,init

def find_scalar_source(m, name, out2node, init):
    if name in init:
        return ('initializer', name, numpy_helper.to_array(init[name]))
    n = out2node.get(name)
    if n is None:
        for gi in m.graph.input:
            if gi.name==name: return ('input', name, None)
        return ('unknown', name, None)
    if n.op_type=='Constant':
        for a in n.attribute:
            if a.name=='value' and a.type==onnx.AttributeProto.TENSOR:
                return ('constant', n.name, numpy_helper.to_array(a.t))
            if a.name=='value_int':
                return ('constant', n.name, np.array([a.i],np.int64))
    if n.op_type in PASSTHRU and n.input:
        return find_scalar_source(m, n.input[0], out2node, init)
    return ('unknown', n.name, None)

def ensure_input(g, name, dtype, shape):
    if not any(i.name==name for i in g.input):
        g.input.extend([helper.make_tensor_value_info(name, dtype, shape)])

def promote_all_topk_k_to_input(m, input_name='TopK_k'):
    g=m.graph; out2node,init=maps(m)
    ensure_input(g, input_name, TensorProto.INT64, [1])
    rewired=0
    for n in g.node:
        if n.op_type!='TopK' or len(n.input)<2: continue
        src_type,_,_ = find_scalar_source(m, n.input[1], out2node, init)
        if src_type in ('initializer','constant'):
            n.input[1]=input_name; rewired+=1
    print(f'[TopK] rewired {rewired} nodes to input "{input_name}"')
    return m

def promote_all_nms_thresholds_to_inputs(m, iou_name='NMS_iou_thr', score_name='NMS_score_thr'):
    g=m.graph; out2node,init=maps(m)
    # ONNX NMS: inputs = [boxes, scores, max_per_class?, iou_thr?, score_thr?]
    # 我们只把第4/5个（阈值）改成输入（若存在且是常量）
    ensure_input(g, iou_name,   TensorProto.FLOAT, [1])
    ensure_input(g, score_name, TensorProto.FLOAT, [1])
    rewired_iou=rewired_sc=0
    for n in g.node:
        if n.op_type!='NonMaxSuppression': continue
        # iou_thr 在 input[3]
        if len(n.input)>=4 and n.input[3]:
            st,_,_=find_scalar_source(m, n.input[3], out2node, init)
            if st in ('initializer','constant'):
                n.input[3]=iou_name; rewired_iou+=1
        # score_thr 在 input[4]
        if len(n.input)>=5 and n.input[4]:
            st,_,_=find_scalar_source(m, n.input[4], out2node, init)
            if st in ('initializer','constant'):
                n.input[4]=score_name; rewired_sc+=1
    print(f'[NMS] rewired iou:{rewired_iou} score:{rewired_sc}')
    return m

if __name__=='__main__':
    onnx_in  = 'model.onnx'
    onnx_out = 'model_runtime_k.onnx'
    m = load(onnx_in)
    m = promote_all_topk_k_to_input(m, input_name='TopK_k')
    # 可选：如果模型里有 NMS 且你想动态阈值，也打开这一行：
    # m = promote_all_nms_thresholds_to_inputs(m, iou_name='NMS_iou_thr', score_name='NMS_score_thr')
    save(m, onnx_out)
    print('saved ->', onnx_out)

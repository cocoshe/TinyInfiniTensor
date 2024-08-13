#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/transpose.h"
#include "operators/matmul.h"

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        if (ops.empty()) return;
        for (size_t i = 0; i < ops.size(); i ++) {
            auto& op = ops[i];
            if (op->getOpType() == OpType::Transpose) {
                for (auto& next_op: op->getSuccessors()) {
                    if (next_op->getOpType() == OpType::Transpose) {
                        auto&& perm1 = std::dynamic_pointer_cast<TransposeObj>(op)->getPermute();
                        auto&& perm2 = std::dynamic_pointer_cast<TransposeObj>(next_op)->getPermute();
                        if (perm1 != perm2) continue;
                        i = -1;
                        auto& op_in = op->getInputs()[0];
                        auto& op_out = op->getOutputs()[0];
                        auto& next_op_in = next_op->getInputs()[0];
                        auto& next_op_out = next_op->getOutputs()[0];

                        op_in->removeTarget(op);
                        for (auto& next_next_op: next_op->getSuccessors()) {
                            op_in->addTarget(next_next_op);
                            next_next_op->removePredecessors(next_op);
                            next_next_op->replaceInput(next_op_out, op_in);
                            for (auto& pre_op: op->getPredecessors()) {
                                // pre_op --> op(transpose) --> next_op(transpose) --> next_next_op
                                pre_op->removeSuccessors(op);
                                pre_op->addSuccessors(next_next_op);
                                next_next_op->addPredecessors(pre_op);
                            }
                        }
                        removeTensor(op_out), removeTensor(next_op_out);
                        removeOperator(op), removeOperator(next_op);
                    } 
                }
            } else if (op->getOpType() == OpType::MatMul) {
                // transpose_op1 ------↓
                //                  matmul_op
                // transpose_op2 ------↑
                auto&& matmul_op = std::dynamic_pointer_cast<MatmulObj>(op);
                auto& matmul_ins = matmul_op->getInputs();       
                bool is_optimized = false;         
                for (int i = 0; i < 2; i ++ ) {
                    auto& matmul_in = matmul_ins[i];
                    auto&& trans_op = matmul_in->getSource();
                    if (trans_op && trans_op->getOpType() == OpType::Transpose) {
                        auto& trans_in = trans_op->getInputs()[0];
                        auto& trans_out = trans_op->getOutputs()[0];
                        // check perm
                        auto&& perm = std::dynamic_pointer_cast<TransposeObj>(trans_op)->getPermute();
                        int n_perm = perm.size();
                        bool flag = true;
                        for (int i = 0; i < n_perm-2; i ++ ) {
                            if (perm[i] == i) continue;
                            flag = false;
                            break;
                        }
                        if (!(perm[n_perm-1] == n_perm-2 && perm[n_perm-2] == n_perm-1)) flag = false;
                        if (flag) {
                            if (i == 0) matmul_op->setTransA(true);
                            else matmul_op->setTransB(true);
                            trans_in->removeTarget(trans_op);
                            trans_in->addTarget(matmul_op);
                            matmul_op->removePredecessors(trans_op);
                            matmul_op->replaceInput(trans_out, trans_in);
                            removeTensor(trans_out);
                            removeOperator(trans_op);
                            is_optimized = true;
                        }
                    }
                }
                if (is_optimized) {
                    i = -1;
                }
            }
 
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        
        // Use two for loops instead of one, 
        // because `IT_ASSERT(this->ptr == nullptr);` in `Allocator::alloc`,
        // the physical mem shouldn't be allocated when `Allocator::alloc`.
        unordered_map<Tensor, size_t> mp;
        for (auto& tensor: tensors) {
            size_t sz = tensor->getBytes();
            size_t offset = allocator.alloc(sz);
            mp[tensor] = offset;
        }

        auto&& base = allocator.getPtr();
        for (auto& tensor: tensors) {
            auto&& b = make_ref<BlobObj>(runtime, base + mp[tensor]);
            tensor->setDataBlob(b);
        }
        
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini
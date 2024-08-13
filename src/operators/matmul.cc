#include "operators/matmul.h"
#include "utils/operator_utils.h"
namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        Shape a = inputs[0]->getDims(), b = inputs[1]->getDims();
        int n_a = a.size(), n_b = b.size();
        vector<int> aa(n_a-2), bb(n_b-2);
        std::copy(a.begin(), a.begin()+n_a-2, aa.begin());
        std::copy(b.begin(), b.end()+n_b-2, bb.begin());
        int n_aa = aa.size(), n_bb = bb.size();

        Shape ret = infer_broadcast(aa, bb);
        int x1, x2, x3, x4;
        x1 = a[n_a-2], x2 = a[n_a-1], x3 = b[n_b-2], x4 = b[n_b-1];
        int x = transA ? x2: x1, y = transB ? x3: x4;
        ret.push_back(x);
        ret.push_back(y);
        return {{ret}};
    }

} // namespace infini
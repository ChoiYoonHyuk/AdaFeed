// Exact distance-k greedy coloring and a separate verifier for a sorted, simple CSR.
#include <algorithm>
#include <cstdint>
#include <vector>
extern "C" int color_balls(int n, const int32_t* ptr, const int32_t* col,
                           int k, int32_t* color) {
    std::vector<int> queue(n), visited(n,-1), forbidden(n,-1);
    std::fill(color,color+n,-1);
    int used=0;
    for (int root=0;root<n;++root) {
        int begin=0,end=1;
        queue[0]=root;
        visited[root]=root;
        for (int depth=0; depth<k && begin<end; ++depth) {
            const int layer_end=end;
            while(begin<layer_end) {
                const int v=queue[begin++];
                for(int j=ptr[v];j<ptr[v+1];++j) {
                    const int u=col[j];
                    if (visited[u]==root) continue;
                    visited[u]=root;
                    queue[end++]=u;
                    if (u<root) forbidden[color[u]]=root;
                }
            }
        }
        int c=0;
        while(c<used && forbidden[c]==root) ++c;
        if(c==used) ++used;
        color[root]=c;
    }
    return used;
}
extern "C" int64_t verify_balls(int n,const int32_t* ptr,const int32_t* col,
                                int k,const int32_t* color) {
    std::vector<int> q(n), mark(n,-1), distance(n);
    int64_t examined=0;
    for(int root=0;root<n;++root) {
        q[0]=root;mark[root]=root;distance[root]=0;
        int head=0,tail=1;
        while(head<tail) {
            int v=q[head++];
            if(distance[v]==k) continue;
            for(int j=ptr[v];j<ptr[v+1];++j) {
                int u=col[j];
                if(mark[u]==root) continue;
                mark[u]=root;distance[u]=distance[v]+1; q[tail++]=u;
                if(color[u]==color[root]) return -1;
                ++examined;
            }
        }
    }
    return examined;
}

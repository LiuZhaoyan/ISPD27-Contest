// Bounded Van Ginneken-style tree DP. Each call owns its state and is reentrant.
#include <algorithm>
#include <cmath>
#include <vector>
#include <utility>

struct State {
  double cap, required, area, power;
  std::vector<std::pair<int,int>> ops;
};
static bool dominates(const State& a,const State& b) {
  return a.cap<=b.cap && a.required>=b.required && a.area<=b.area &&
    a.power<=b.power && a.ops.size()<=b.ops.size();
}
static std::vector<State> prune(std::vector<State> values,int budget) {
  std::stable_sort(values.begin(),values.end(),[](const State& a,const State& b){
    if(a.cap!=b.cap)return a.cap<b.cap;
    return a.required>b.required;
  });
  std::vector<State> front;
  for(auto& v:values) {
    bool dead=false;
    for(const auto& q:front)if(dominates(q,v)){dead=true;break;}
    if(dead)continue;
    front.erase(std::remove_if(front.begin(),front.end(),[&](const State& q){return dominates(v,q);}),front.end());
    front.push_back(std::move(v));
  }
  if((int)front.size()<=budget)return front;
  // Preserve objective extremes and deterministic quantiles of the frontier.
  std::vector<int> chosen{0,(int)front.size()-1};
  auto best=[&](auto score){int k=0;for(int i=1;i<(int)front.size();++i)if(score(front[i])<score(front[k]))k=i;return k;};
  chosen.push_back(best([](const State& s){return -s.required;}));
  chosen.push_back(best([](const State& s){return s.area;}));
  chosen.push_back(best([](const State& s){return s.power;}));
  for(int j=0;j<budget;++j)chosen.push_back(j*(front.size()-1)/(budget-1));
  std::vector<State> out;std::vector<bool> used(front.size(),false);
  for(int j:chosen)if(!used[j]&&(int)out.size()<budget){used[j]=true;out.push_back(std::move(front[j]));}
  return out;
}
static double interp(const double* x,const double* y,int n,double load) {
  int i=std::upper_bound(x,x+n,load)-x-1;i=std::max(0,std::min(n-2,i));
  return y[i]+(load-x[i])*(y[i+1]-y[i])/(x[i+1]-x[i]);
}
extern "C" int workflow_buffer_dp(int n,const int* parent,const double* r,
    const double* cap,const double* required,int root,int nb,int nt,
    const double* input_cap,const double* area,const double* power,
    const double* loads,const double* delays,int budget,int max_buffers,
    double driver_r,double area_weight,double power_weight,
    double* scores,double* summary,int* counts,int* nodes,int* buffers) {
  if(n<1||budget<5||max_buffers<1||nt<2)return -1;
  std::vector<std::vector<int>> children(n);
  for(int i=0;i<n;++i)if(parent[i]>=0)children[parent[i]].push_back(i);
  std::vector<int> order{root};
  for(size_t j=0;j<order.size();++j)for(int c:children[order[j]])order.push_back(c);
  if((int)order.size()!=n)return -2;
  std::vector<std::vector<State>> dp(n);
  for(auto it=order.rbegin();it!=order.rend();++it) {
    int node=*it;
    std::vector<State> f{{cap[node],required[node],0.,0.,{}}};
    for(int child:children[node]) {
      std::vector<State> joined;
      for(const auto& a:f)for(const auto& b:dp[child]) {
        if((int)(a.ops.size()+b.ops.size())>max_buffers)continue;
        State s{a.cap+b.cap,std::min(a.required,b.required-r[child]*b.cap),
                a.area+b.area,a.power+b.power,a.ops};
        s.ops.insert(s.ops.end(),b.ops.begin(),b.ops.end());joined.push_back(std::move(s));
      }
      f=prune(std::move(joined),budget);
    }
    std::vector<State> expanded=f;
    for(const auto& s:f)if((int)s.ops.size()<max_buffers)for(int b=0;b<nb;++b) {
      if(!std::isfinite(s.required))continue;
      State v{input_cap[b],s.required-interp(loads+b*nt,delays+b*nt,nt,s.cap),
              s.area+area[b],s.power+power[b],s.ops};
      v.ops.emplace_back(node,b);expanded.push_back(std::move(v));
    }
    dp[node]=prune(std::move(expanded),budget);
  }
  auto objective=[&](const State& s){return s.required-driver_r*s.cap-area_weight*s.area-power_weight*s.power;};
  auto& f=dp[root];
  std::stable_sort(f.begin(),f.end(),[&](const State& a,const State& b){return objective(a)>objective(b);});
  for(int i=0;i<(int)f.size();++i) {
    scores[i]=objective(f[i]);counts[i]=f[i].ops.size();
    summary[4*i]=f[i].cap;summary[4*i+1]=f[i].required;summary[4*i+2]=f[i].area;summary[4*i+3]=f[i].power;
    for(int j=0;j<counts[i];++j){nodes[i*max_buffers+j]=f[i].ops[j].first;buffers[i*max_buffers+j]=f[i].ops[j].second;}
  }
  return f.size();
}

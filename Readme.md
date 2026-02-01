## This is the base format for kube + kustomize + github-ci
1. Docker build
```
Docker build -f Dockerfile.mysql -t my-mysql:latest . 
Docker build -f Dockerfile.redis -t my-redis:latest . 
Docker build -f Dockerfile.mongo -t my-mongo:latest . 
Docker build -f Dockerfile.learner -t my-learner:latest . 
Docker build -f Dockerfile.actor -t my-actor:latest . 
```

2. create ingress controller
```
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.10.0/deploy/static/provider/cloud/deploy.yaml
```
3. run kustomize
```
kubectl create namespace argocd
kubectl apply -k argocd-kusto
kubectl create namespace mykube
kubectl apply -k mykube-kusto
```
4. ArgoCD
https://localhost/argocd
4. kubectl dashboard
```
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml
kubectl create serviceaccount dashboard-admin-user -n kubernetes-dashboard
kubectl create clusterrolebinding dashboard-admin-user --clusterrole=cluster-admin --serviceaccount=kubernetes-dashboard:dashboard-admin-user
kubectl proxy
kubectl -n kubernetes-dashboard create token dashboard-admin-user
http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/
```

## This is the base format for kube + kustomize + github-ci
1. create namespace
```
kubectl get pods -n myargocd
```
2. run kustomize
```
kubectl apply -k .
```

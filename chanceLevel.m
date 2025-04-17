function [CL] = chanceLevel(labels,N)

classes = unique(labels);
acc = zeros(N,1);
for i =1:N
    rp = randperm(size(labels,1));
    randPred = zeros(size(labels))+classes(2);
    randPred(rp') = classes(1);
    acc(i) = sum(randPred==labels)/size(labels,1);
end

CL = mean(acc);

end
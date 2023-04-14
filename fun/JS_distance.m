function dist=JS_distance(X)
    [row,N]=size(X);
    dist=zeros(N,N);
    M=zeros(row,1);
    for i=1:N
        for j=1:N
            M=0.5*(X(:,i)+X(:,j));
            dist(i,j)=0.5*X(:,i)'*log(X(:,i)./M)+0.5*X(:,j)'*log(X(:,j)./M);          
        end
    end

end
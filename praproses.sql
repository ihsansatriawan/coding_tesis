drop table point_receivedfrom_from_hijup_2015;
create table point_receivedfrom_from_hijup_2015 as
select * from point_histories 
where 
  category = 'Received From' and
  date(created_at+INTERVAL'7 hour') >= '2015-01-01' and
  detail like 'FREE%'

drop table point_redeem_from_2015;
create table point_redeem_from_2015 as
select * from point_histories 
where 
  category = 'Redeem Points' and
  date(created_at+INTERVAL'7 hour') >= '2015-01-01'


drop table point_transferto_from_2015;
create table point_transferto_from_2015 as
select 
  * 
from 
  point_histories 
where 
  category = 'Transfer To' and 
  owner_type != 'Admin' and
  date(created_at+INTERVAL'7 hour') >= '2015-01-01'


drop table point_receivedfrom_from_2015;
create table point_receivedfrom_from_2015 as
select * from point_histories 
where 
  category = 'Received From' and
  date(created_at+INTERVAL'7 hour') >= '2015-01-01'

create table point_receivedfrom_from_nohijup_2015
select * from point_histories 
where 
  category = 'Received From' and
  date(created_at+INTERVAL'7 hour') >= '2015-01-01' and
  detail not like 'FREE%'


select
  id_pengirim,
  id_penerima,
  count(*) as freq
from
(
  select 
    a.id as id_pengirim, b.owner_id as id_penerima, b.point
  from 
    users a
  inner join
    point_receivedfrom_from_nohijup_2015 b
      on a.email = b.from
) as foo
group by id_pengirim, id_penerima

select
  id_pengirim,
  id_penerima,
  count(*) as freq
from
(
  select 
    b.owner_id as id_pengirim, a.id as id_penerima, ABS(b.point)
  from 
    users a
  inner join
    point_transferto_from_2015 b
      on a.email = b.to
) as foo
group by id_pengirim, id_penerima
